#include "Assembler/ELFGenerator.h"

#include <sstream>
#include <unordered_map>

#include "Libraries/elfio/elfio.hpp"
#include "Libraries/elfio/elfio_dump.hpp"
#include "Libraries/wmemstreambuf.hpp"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

#define ELFABI_NVIDIA_VERSION 7
#define ELF_VERSION 110
#define ELF_SREG_SIZE 0x140

#define EF_CUDA_SM(x) (x)
#define EF_CUDA_VIRTUAL_SM(x) (x << 16)
#define EF_CUDA_TEXMODE_UNIFIED (0x1 << 8)
#define EF_CUDA_64BIT_ADDRESS (0x4 << 8)

#define STO_CUDA_ENTRY 0x10
#define STO_CUDA_CONSTANT 0x80
#define STT_CUDA_OBJECT 0xd

#define SHI_REGISTERS(x) (x << 24)
#define SHF_BARRIERS(x) (x << 20)

#define R_CUDA_ABS32_HI_20 0x2c
#define R_CUDA_ABS32_LO_20 0x2b

#define SZ_SHORT 2
#define SZ_WORD 4

namespace Assembler {

template<class T>
std::vector<char> ELFGenerator::DecomposeShort(const T& value)
{
	auto value16 = static_cast<std::uint16_t>(value);
	auto charArray = reinterpret_cast<char *>(&value16);
	return {charArray[0], charArray[1]};
}

template<class T>
std::vector<char> ELFGenerator::DecomposeWord(const T& value)
{
	auto value64 = static_cast<std::uint64_t>(value);
	auto charArray = reinterpret_cast<char *>(&value64);
	return {charArray[0], charArray[1], charArray[2], charArray[3]};
}

void ELFGenerator::AppendBytes(std::vector<char>& buffer, const std::vector<char>& bytes)
{
	buffer.insert(std::end(buffer), std::begin(bytes), std::end(bytes));
}

ELFBinary *ELFGenerator::Generate(const BinaryProgram *program)
{
	auto timeELF_start = Utils::Chrono::Start("ELF generator");

	// Initialize properties of ELF file

	ELFIO::elfio writer;
	writer.create(ELFCLASS64, ELFDATA2LSB);
	writer.set_os_abi(ELFOSABI_NVIDIA);
	writer.set_abi_version(ELFABI_NVIDIA_VERSION);
	writer.set_version(ELF_VERSION);
	writer.set_type(ET_REL);
	writer.set_machine(EM_CUDA);
	writer.set_flags(
		EF_CUDA_VIRTUAL_SM(program->GetComputeCapability()) |
		EF_CUDA_SM(program->GetComputeCapability()) |
		EF_CUDA_TEXMODE_UNIFIED |
		EF_CUDA_64BIT_ADDRESS
	);
	writer.set_entry(0x0);

	// Initialize string and symbol table sections

	auto stringSection = writer.sections.add(".strtab");
	stringSection->set_type(SHT_STRTAB);
	stringSection->set_addr_align(0x1);

	auto symbolSection = writer.sections.add(".symtab");
	symbolSection->set_type(SHT_SYMTAB);
	symbolSection->set_link(stringSection->get_index());
	symbolSection->set_addr_align(0x8);
	symbolSection->set_entry_size(writer.get_default_entry_size(SHT_SYMTAB));

	ELFIO::string_section_accessor stringWriter(stringSection);
	ELFIO::symbol_section_accessor symbolWriter(writer, symbolSection);

	// Custom NVIDIA info section with memory limits (regs, stack, frame) for each function

	auto infoSection = writer.sections.add(".nv.info");
	infoSection->set_type(SHT_LOPROC);
	infoSection->set_addr_align(0x4);
	infoSection->set_link(symbolSection->get_index());

	// Construct segments for text and shared data sections

	auto phdrSegment = writer.segments.add();
	phdrSegment->set_type(PT_PHDR);
	phdrSegment->set_flags(PF_X | PF_R);
	phdrSegment->set_align(0x8);

	auto textSegment = writer.segments.add();
	textSegment->set_type(PT_LOAD);
	textSegment->set_flags(PF_X | PF_R);
	textSegment->set_align(0x8);

	auto dataSegment = writer.segments.add();
	dataSegment->set_type(PT_LOAD);
	dataSegment->set_flags(PF_R | PF_W);
	dataSegment->set_align(0x8);

	// Maintain a symbol offset for connecting the .text to the symbol (NVIDIA requirement)

	auto symbolCount = 0u;
	auto symbolOffset = 0u;

	// Global variables

	std::unordered_map<std::string, ELFIO::Elf_Word> globalSymbolMap;
	if (program->GetGlobalVariableCount() > 0)
	{
		// Add NVIDIA global section that will contain all variables

		auto globalSection = writer.sections.add(".nv.global");
		globalSection->set_type(SHT_NOBITS);
		globalSection->set_flags(SHF_ALLOC | SHF_WRITE);
		globalSection->set_addr_align(0x4); //TODO: Alignment depends on variables

		dataSegment->add_section_index(globalSection->get_index(), globalSection->get_addr_align());

		symbolWriter.add_symbol(stringWriter,
			".nv.global", 0x00000000, 0, STB_LOCAL, STT_SECTION, STV_DEFAULT, globalSection->get_index()
		);

		// Add a symbol object for each global variable

		std::size_t totalSize = 0;
		for (const auto& [name, offset, size] : program->GetGlobalVariables())
		{
			auto index = symbolWriter.add_symbol(stringWriter,
				name.c_str(), offset, size, STB_LOCAL, STT_OBJECT, STV_DEFAULT, globalSection->get_index()
			);
			globalSymbolMap[name] = index;
			totalSize += size;
		}
		globalSection->set_size(totalSize);

		symbolCount += program->GetGlobalVariableCount() + 1;
		symbolOffset += program->GetGlobalVariableCount() + 1;
	}

	// Info buffer for accumulating properties

	std::vector<char> infoBuffer;

	// Add functions .data and .text sections

	for (const auto& function : program->GetFunctions())
	{
		auto allocateSharedMemory = (program->GetDynamicSharedMemory() || function->GetSharedMemorySize() > 0);
		auto allocateConstantMemory = (function->GetConstantMemorySize() > 0);

		auto textSymbol = symbolOffset + 4 + allocateSharedMemory + allocateConstantMemory;
		auto dataSymbol = symbolOffset + 2 + allocateSharedMemory;

		// Add custom NVIDIA info

		// EIATTR_REGCOUNT
		//     Format: EIFMT_SVAL
		//     Value:  function: func(0x5)      register count: 7
		//
		//     /*0000*/        .byte   0x04, 0x2f
		//     /*0002*/        .short  (.L_2 - .L_1)
		// .L_1:
		//     /*0004*/        .word   index@(func)
		//     /*0008*/        .word   0x00000007
		// .L_2:

		AppendBytes(infoBuffer, {(char)Type::EIFMT_SVAL});
		AppendBytes(infoBuffer, {(char)Attribute::EIATTR_REG_COUNT});
		AppendBytes(infoBuffer, DecomposeShort(2*SZ_WORD));               // Size
		AppendBytes(infoBuffer, DecomposeWord(textSymbol));               // Function index
		AppendBytes(infoBuffer, DecomposeWord(function->GetRegisters())); // Value

		// EIATTR_MAX_STACK_SIZE
		//     Format: EIFMT_SVAL
		//     Value:  function: func(0x5)      stack size: 0x0
		//
		//     /*0000*/        .byte   0x04, 0x23
		//     /*0002*/        .short  (.L_2 - .L_1)
		// .L_1:
		//     /*0004*/        .word   index@(func)
		//     /*0008*/        .word   0x00000000
		// .L_2:

		AppendBytes(infoBuffer, {(char)Type::EIFMT_SVAL});
		AppendBytes(infoBuffer, {(char)Attribute::EIATTR_MAX_STACK_SIZE});
		AppendBytes(infoBuffer, DecomposeShort(2*SZ_WORD)); // Size
		AppendBytes(infoBuffer, DecomposeWord(textSymbol)); // Function index
		AppendBytes(infoBuffer, DecomposeWord(0));          // Value

		// EIATTR_MIN_STACK_SIZE
		//     Format: EIFMT_SVAL
		//     Value:  function: func(0x5)      stack size: 0x0
		//
		//     /*0000*/        .byte   0x04, 0x12
		//     /*0002*/        .short  (.L_2 - .L_1)
		// .L_1:
		//     /*0004*/        .word   index@(func)
		//     /*0008*/        .word   0x00000000
		// .L_2:

		AppendBytes(infoBuffer, {(char)Type::EIFMT_SVAL});
		AppendBytes(infoBuffer, {(char)Attribute::EIATTR_MIN_STACK_SIZE});
		AppendBytes(infoBuffer, DecomposeShort(2*SZ_WORD)); // Size
		AppendBytes(infoBuffer, DecomposeWord(textSymbol)); // Function index
		AppendBytes(infoBuffer, DecomposeWord(0));          // Value

		// EIATTR_FRAME_SIZE
		//     Format: EIFMT_SVAL
		//     Value:  function: func(0x5)      frame size: 0x0
		//
		//     /*0000*/        .byte   0x04, 0x11
		//     /*0002*/        .short  (.L_2 - .L_1)
		// .L_1:
		//     /*0004*/        .word   index@(func)
		//     /*0008*/        .word   0x00000000
		// .L_2:

		AppendBytes(infoBuffer, {(char)Type::EIFMT_SVAL});
		AppendBytes(infoBuffer, {(char)Attribute::EIATTR_FRAME_SIZE});
		AppendBytes(infoBuffer, DecomposeShort(2*SZ_WORD)); // Size
		AppendBytes(infoBuffer, DecomposeWord(textSymbol)); // Function index
		AppendBytes(infoBuffer, DecomposeWord(0));          // Value

		// Custom NVIDIA info section for function properties (params, exits, s2rctaid)

		auto functionInfoSection = writer.sections.add(".nv.info." + function->GetName());
		functionInfoSection->set_type(SHT_LOPROC);
		functionInfoSection->set_addr_align(0x4);
		functionInfoSection->set_link(symbolSection->get_index());

		std::vector<char> functionInfoBuffer;

		// EIATTR_SW2393858_WAR
		//     Format: EIFMT_NVAL
		// 
		//     /*0000*/        .byte   0x01, 0x30
		//     .zero           2

		AppendBytes(functionInfoBuffer, {(char)Type::EIFMT_NVAL});
		AppendBytes(functionInfoBuffer, {(char)Attribute::EIATTR_SW2393858_WAR});
		AppendBytes(functionInfoBuffer, {0, 0}); // Zero

		// EIATTR_SW1850030_WAR
		//     Format: EIFMT_NVAL
		//
		//     /*0004*/        .byte   0x01, 0x2a
		//     .zero           2

		AppendBytes(functionInfoBuffer, {(char)Type::EIFMT_NVAL});
		AppendBytes(functionInfoBuffer, {(char)Attribute::EIATTR_SW1850030_WAR});
		AppendBytes(functionInfoBuffer, {0, 0}); // Zero

		if (function->GetParametersCount() > 0)
		{
			// EIATTR_PARAM_CBANK
			//     Format: EIFMT_SVAL
			//     Value:  0x2 0x100140
			//
			//     /*0008*/        .byte   0x04, 0x0a
			//     /*000a*/        .short  (.L_2 - .L_1)
			// .L_1:
			//     /*000c*/        .word   index@(.constant.func)
			//     /*0010*/        .short  0x0140
			//     /*0012*/        .short  0x0010
			// .L_2:

			AppendBytes(functionInfoBuffer, {(char)Type::EIFMT_SVAL});
			AppendBytes(functionInfoBuffer, {(char)Attribute::EIATTR_PARAM_CBANK});
			AppendBytes(functionInfoBuffer, DecomposeShort(SZ_WORD + 2*SZ_SHORT));          // Size
			AppendBytes(functionInfoBuffer, DecomposeWord(dataSymbol));                     // Data section index
			AppendBytes(functionInfoBuffer, DecomposeShort(ELF_SREG_SIZE));                 // Param offset
			AppendBytes(functionInfoBuffer, DecomposeShort(function->GetParametersSize())); // Param size

			// EIATTR_CBANK_PARAM_SIZE
			//     Format: EIFMT_HVAL
			//     Value:  0x10
			//
			//     /*0000*/        .byte   0x03, 0x19
			//     /*0004*/        .short  0x0010

			AppendBytes(functionInfoBuffer, {(char)Type::EIFMT_HVAL});
			AppendBytes(functionInfoBuffer, {(char)Attribute::EIATTR_CBANK_PARAM_SIZE});
			AppendBytes(functionInfoBuffer, DecomposeShort(function->GetParametersSize())); // Param size

			// EIATTR_KPARAM_INFO
			//     Format: EIFMT_SVAL
			//     Value:  Index : 0x0     Ordinal : 0x1   Offset  : 0x8   Size    : 0x8
			//             Pointee's logAlignment : 0x0    Space : 0x0     cbank : 0x1f    Parameter Space : CBANK
			//
			//     /*0018*/        .byte   0x04, 0x17
			//     /*001a*/        .short  (.L_2 - .L_1)
			// .L_1:
			//     /*001c*/        .word   0x00000000
			//     /*0020*/        .short  0x0001
			//     /*0022*/        .short  0x0008
			//     /*0024*/        .byte   0x00, 0xf0, 0x21, 0x00
			// .L_2:

			auto paramIndex = 0u;
			auto paramOffset = 0u;
			for (const auto& parameter : function->GetParameters())
			{
				AppendBytes(functionInfoBuffer, {(char)Type::EIFMT_SVAL});
				AppendBytes(functionInfoBuffer, {(char)Attribute::EIATTR_KPARAM_INFO});
				AppendBytes(functionInfoBuffer, DecomposeShort(2*SZ_WORD + 2*SZ_SHORT)); // Size
				AppendBytes(functionInfoBuffer, DecomposeWord(0));                       // Index
				AppendBytes(functionInfoBuffer, DecomposeShort(paramIndex));             // Parameter info
				AppendBytes(functionInfoBuffer, DecomposeShort(paramOffset));            // Parameter offset
				AppendBytes(functionInfoBuffer, DecomposeWord(
					0x00            << 28 | // Log align
					(parameter / 4) << 20 | // Size
					0x1f            << 12 | // cbank
					0x000           << 0    // Space
				));

				paramIndex++;
				paramOffset += parameter;
			}
		}

		// EIATTR_MAXREG_COUNT
		//     Format: EIFMT_HVAL
		//     Value:  0xff
		//
		//     /*0000*/        .byte   0x03, 0x1b
		//     /*0002*/        .short  0x00ff

		AppendBytes(functionInfoBuffer, {(char)Type::EIFMT_HVAL});
		AppendBytes(functionInfoBuffer, {(char)Attribute::EIATTR_MAXREG_COUNT});
		AppendBytes(functionInfoBuffer, DecomposeShort(255)); // Value

		if (auto count = function->GetS2RCTAIDOffsetsCount(); count > 0)
		{
			// EIATTR_S2RCTAID_INSTR_OFFSETS
			//     Format: EIFMT_SVAL
			//     Value:  0x10
			//
			//     /*0000*/        .byte   0x04, 0x1d
			//     /*0002*/        .short  (.L_2 - .L_1)
			// .L_1:
			//     /*0004*/        .word   0x00000010
			// .L_2:

			AppendBytes(functionInfoBuffer, {(char)Type::EIFMT_SVAL});
			AppendBytes(functionInfoBuffer, {(char)Attribute::EIATTR_S2RCTAID_INSTR_OFFSETS});
			AppendBytes(functionInfoBuffer, DecomposeShort(count * SZ_WORD)); // Size

			for (const auto& ctaOffset : function->GetS2RCTAIDOffsets())
			{
				AppendBytes(functionInfoBuffer, DecomposeWord(ctaOffset));
			}
		}

		if (auto count = function->GetExitOffsetsCount(); count > 0)
		{
			// EIATTR_EXIT_INSTR_OFFSETS
			//     Format: EIFMT_SVAL
			//     Value:  0x10
			//
			//     /*0000*/        .byte   0x04, 0x1c
			//     /*0002*/        .short  (.L_2 - .L_1)
			// .L_1:
			//     /*0004*/        .word   0x00000010
			// .L_2:

			AppendBytes(functionInfoBuffer, {(char)Type::EIFMT_SVAL});
			AppendBytes(functionInfoBuffer, {(char)Attribute::EIATTR_EXIT_INSTR_OFFSETS});
			AppendBytes(functionInfoBuffer, DecomposeShort(count * SZ_WORD)); // Size

			for (const auto& exitOffset : function->GetExitOffsets())
			{
				AppendBytes(functionInfoBuffer, DecomposeWord(exitOffset));
			}
		}

		if (auto count = function->GetCoopOffsetsCount(); count > 0)
		{
			// EIATTR_COOP_GROUP_INSTR_OFFSETS
			//     Format: EIFMT_SVAL
			//     Value:  0x10
			//
			//     /*0000*/        .byte   0x04, 0x28
			//     /*0002*/        .short  (.L_2 - .L_1)
			// .L_1:
			//     /*0004*/        .word   0x00000010
			// .L_2:

			AppendBytes(functionInfoBuffer, {(char)Type::EIFMT_SVAL});
			AppendBytes(functionInfoBuffer, {(char)Attribute::EIATTR_COOP_GROUP_INSTR_OFFSETS});
			AppendBytes(functionInfoBuffer, DecomposeShort(count * SZ_WORD)); // Size

			for (const auto& coopOffset : function->GetCoopOffsets())
			{
				AppendBytes(functionInfoBuffer, DecomposeWord(coopOffset));
			}
		}

		if (auto [dimX, dimY, dimZ] = function->GetRequiredThreads(); dimX > 0)
		{
			// EIATTR_REQNTID
			//     Format: EIFMT_SVAL
			//     Value:  0x400 0x1 0x1
			//
			//     /*0000*/        .byte   0x04, 0x10
			//     /*0002*/        .short  (.L_28 - .L_27)
			// .L_1:
			//     /*0004*/        .word   0x00000400
			//     /*0008*/        .word   0x00000001
			//     /*000c*/        .word   0x00000001
			// .L_2:

			AppendBytes(functionInfoBuffer, {(char)Type::EIFMT_SVAL});
			AppendBytes(functionInfoBuffer, {(char)Attribute::EIATTR_REQNTID});
			AppendBytes(functionInfoBuffer, DecomposeShort(3*SZ_WORD)); // Size
			AppendBytes(functionInfoBuffer, DecomposeWord(dimX));       // Dimension X
			AppendBytes(functionInfoBuffer, DecomposeWord(dimY));       // Dimension Y
			AppendBytes(functionInfoBuffer, DecomposeWord(dimZ));       // Dimension Z
		}
		else if (auto [dimX, dimY, dimZ] = function->GetMaxThreads(); dimX > 0)
		{
			// EIATTR_MAX_THREADS
			//     Format: EIFMT_SVAL
			//     Value: 0x400 0x1 0x1
			//
			//     /*0000*/        .byte   0x04, 0x05
			//     /*0002*/        .short  (.L_28 - .L_27)
			// .L_1:
			//     /*0004*/        .word   0x00000400
			//     /*0008*/        .word   0x00000001
			//     /*000c*/        .word   0x00000001
			// .L_2:

			AppendBytes(functionInfoBuffer, {(char)Type::EIFMT_SVAL});
			AppendBytes(functionInfoBuffer, {(char)Attribute::EIATTR_MAX_THREADS});
			AppendBytes(functionInfoBuffer, DecomposeShort(3*SZ_WORD)); // Size
			AppendBytes(functionInfoBuffer, DecomposeWord(dimX));       // Dimension X
			AppendBytes(functionInfoBuffer, DecomposeWord(dimY));       // Dimension Y
			AppendBytes(functionInfoBuffer, DecomposeWord(dimZ));       // Dimension Z
		}

		if (function->GetCTAIDZUsed())
		{
			// EIATTR_CTAIDZ_USED
			//     Format: EIFMT_NVAL
			//
			//     /*0080*/        .byte   0x01, 0x04
			//     .zero           2

			AppendBytes(functionInfoBuffer, {(char)Type::EIFMT_NVAL});
			AppendBytes(functionInfoBuffer, {(char)Attribute::EIATTR_CTAIDZ_USED});
			AppendBytes(functionInfoBuffer, {0, 0}); // Zero
		}

		if (auto count = function->GetIndirectBranchesCount(); count > 0)
		{
			// EIATTR_INDIRECT_BRANCH_TARGETS
			//     Format: EIFMT_SVAL
			// 
			//     Offset of Indirect Branch: 0x48  Number of targets: 1
			//     Targets: 0x70
			//
			//     /*0000*/        .byte   0x04, 0x34
			//     /*0002*/        .short  (.L_2 - .L_1)
			// .L_1:
			//     /*0004*/        .word   .L_4@srel   (offset=0x48)
			//     /*0008*/        .short  0x0         (?)
			//     /*000a*/        .short  0x0         (?)
			//     /*000c*/        .word   0x1         (target count=1)
			//     /*0010*/        .word   .L_3@srel   (target=0x70)
			// .L_2:

			AppendBytes(functionInfoBuffer, {(char)Type::EIFMT_SVAL});
			AppendBytes(functionInfoBuffer, {(char)Attribute::EIATTR_INDIRECT_BRANCH_TARGETS});
			AppendBytes(functionInfoBuffer, DecomposeShort(count*(3*SZ_WORD+2*SZ_SHORT)));

			for (const auto& [offset, target] : function->GetIndirectBranches())
			{
				AppendBytes(functionInfoBuffer, DecomposeWord(offset)); // Offset
				AppendBytes(functionInfoBuffer, DecomposeShort(0));     // ?
				AppendBytes(functionInfoBuffer, DecomposeShort(0));     // ?
				AppendBytes(functionInfoBuffer, DecomposeWord(1));      // Target count
				AppendBytes(functionInfoBuffer, DecomposeWord(target)); // Target
			}
		}

		functionInfoSection->set_data(functionInfoBuffer.data(), functionInfoBuffer.size());

		// Add constant data section:
		//   - 0x140 base
		//   - Space for each parameter

		auto dataSize = ELF_SREG_SIZE + function->GetParametersSize();
		auto data = new char[dataSize](); // Zero initialized

		auto dataSection = writer.sections.add(".nv.constant0." + function->GetName());
		dataSection->set_type(SHT_PROGBITS);
		dataSection->set_flags(SHF_ALLOC);
		dataSection->set_addr_align(0x4);
		dataSection->set_data(data, dataSize);

		textSegment->add_section_index(dataSection->get_index(), dataSection->get_addr_align());

		// Add constant variables section

		ELFIO::section *constSection = nullptr;
		if (allocateConstantMemory)
		{
			// Copy constant data

			auto constSize = function->GetConstantMemorySize();
			auto constData = new char[constSize]();
			std::memcpy(constData, function->GetConstantMemory().data(), constSize * sizeof(char));

			// Allocate section

			constSection = writer.sections.add(".nv.constant2." + function->GetName());
			constSection->set_type(SHT_PROGBITS);
			constSection->set_flags(SHF_ALLOC);
			constSection->set_addr_align(0x4);
			constSection->set_data(constData, constSize);

			textSegment->add_section_index(constSection->get_index(), constSection->get_addr_align());
		}

		// Add .text section for the function body

		auto textSection = writer.sections.add(".text." + function->GetName());
		textSection->set_type(SHT_PROGBITS);
		textSection->set_flags(SHF_BARRIERS(function->GetBarriers()) | SHF_ALLOC | SHF_EXECINSTR);
		textSection->set_addr_align(0x20);
		textSection->set_link(symbolSection->get_index());
		textSection->set_info(SHI_REGISTERS(function->GetRegisters()) + textSymbol);
		textSection->set_data(function->GetText(), function->GetSize());

		textSegment->add_section_index(textSection->get_index(), textSection->get_addr_align());

		// Add shared variables section

		ELFIO::section *sharedSection = nullptr;
		if (allocateSharedMemory)
		{
			sharedSection = writer.sections.add(".nv.shared." + function->GetName());
			sharedSection->set_type(SHT_NOBITS);
			sharedSection->set_flags(SHF_WRITE | SHF_ALLOC);
			sharedSection->set_addr_align(0x10);
			sharedSection->set_info(textSection->get_index());
			sharedSection->set_size(function->GetSharedMemorySize());

			dataSegment->add_section_index(sharedSection->get_index(), sharedSection->get_addr_align());
		}

		// Link info/data sections to .text

		dataSection->set_info(textSection->get_index());
		functionInfoSection->set_info(textSection->get_index());
		if (constSection != nullptr)
		{
			constSection->set_info(textSection->get_index());
		}

		// Create relocation section for .text

		if (function->GetRelocationsCount() > 0)
		{
			auto relocationSection = writer.sections.add(".rel.text." + function->GetName());
			relocationSection->set_type(SHT_REL);
			relocationSection->set_info(textSection->get_index());
			relocationSection->set_link(symbolSection->get_index());
			relocationSection->set_addr_align(0x4);
			relocationSection->set_entry_size(writer.get_default_entry_size(SHT_REL));

			ELFIO::relocation_section_accessor relocationWriter(writer, relocationSection);
			for (const auto& [name, address, kind] : function->GetRelocations())
			{
				auto symbolIndex = globalSymbolMap.at(name);
				if (kind == BinaryFunction::RelocationKind::ABS32_LO_20)
				{
					relocationWriter.add_entry(address, symbolIndex, (unsigned char)R_CUDA_ABS32_LO_20);
				}
				else if (kind == BinaryFunction::RelocationKind::ABS32_HI_20)
				{
					relocationWriter.add_entry(address, symbolIndex, (unsigned char)R_CUDA_ABS32_HI_20);
				}
			}
		}

		// Update symbol table:
		//   - .text
		//  [- .nv.shared]
		//   - .nv.constant0
		//  [- .nv.constant2]
		//   - _param
		//   - Global .text symbol

		symbolWriter.add_symbol(stringWriter,
			textSection->get_name().c_str(), 0x00000000, 0, STB_LOCAL, STT_SECTION, STV_DEFAULT, textSection->get_index()
		);
		if (sharedSection != nullptr)
		{
			symbolWriter.add_symbol(stringWriter,
				sharedSection->get_name().c_str(), 0x00000000, 0, STB_LOCAL, STT_SECTION, STV_DEFAULT, sharedSection->get_index()
			);
			symbolOffset++;
			symbolCount++;
		}
		symbolWriter.add_symbol(stringWriter,
			dataSection->get_name().c_str(), 0x00000000, 0, STB_LOCAL, STT_SECTION, STV_DEFAULT, dataSection->get_index()
		);
		if (constSection != nullptr)
		{
			symbolWriter.add_symbol(stringWriter,
				constSection->get_name().c_str(), 0x00000000, 0, STB_LOCAL, STT_SECTION, STV_DEFAULT, constSection->get_index()
			);
			symbolOffset++;
			symbolCount++;
		}
		symbolWriter.add_symbol(stringWriter,
			"_param", ELF_SREG_SIZE, function->GetParametersSize(), STB_LOCAL, STT_CUDA_OBJECT, STV_INTERNAL| STO_CUDA_CONSTANT, dataSection->get_index()
		);
		symbolWriter.add_symbol(stringWriter,
			function->GetName().c_str(), 0x00000000, textSection->get_size(), STB_GLOBAL, STT_FUNC, STV_DEFAULT | STO_CUDA_ENTRY, textSection->get_index()
		);

		symbolCount += 3; // Does not count global .text section
		symbolOffset += 4;
	}

	// Update the symbol count

	symbolSection->set_info(symbolCount); // NV: Ignores global symbols

	// Set full info, contains data for all functions

	infoSection->set_data(infoBuffer.data(), infoBuffer.size());

	// Print ELF information

	if (Utils::Options::IsBackend_PrintELF())
	{
		Utils::Logger::LogInfo("Asembled ELF file");

		ELFIO::dump::header(std::cout, writer);
		ELFIO::dump::section_headers(std::cout, writer);
		ELFIO::dump::segment_headers(std::cout, writer);
		ELFIO::dump::symbol_tables(std::cout, writer);
		ELFIO::dump::notes(std::cout, writer);
		ELFIO::dump::modinfo(std::cout, writer);
		ELFIO::dump::dynamic_tags(std::cout, writer);
		ELFIO::dump::section_datas(std::cout, writer);
		ELFIO::dump::segment_datas(std::cout, writer);
	}

	// Output ELF binary to memory, copying data

	wmemstreambuf buffer(4096);
	std::ostream binaryStream(&buffer);
	writer.save(binaryStream);

	if (Utils::Options::IsBackend_DumpELF())
	{
		writer.save("r3d3_dump.cubin");
	}

	std::size_t elfSize = 0;
	auto bufferContent = buffer.getcontent(elfSize);

	auto elfBinary = ::operator new(elfSize);
	std::memcpy(elfBinary, bufferContent, elfSize);

	Utils::Chrono::End(timeELF_start);

	if (Utils::Options::IsDebug_Print())
	{
		Utils::Logger::LogDebug("Generate ELF binary, " + std::to_string(elfSize) + " bytes");
	}

	return new ELFBinary(elfBinary, elfSize);
}

}
