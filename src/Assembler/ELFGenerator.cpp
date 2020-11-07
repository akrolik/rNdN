#include "Assembler/ELFGenerator.h"

#include <sstream>

#include "Libraries/elfio/elfio.hpp"
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
	auto timeELF_start = Utils::Chrono::Start("ELF Generator");

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
	symbolSection->set_info(3 * program->GetFunctionCount()); // NV: Ignores global symbols
	symbolSection->set_link(stringSection->get_index());
	symbolSection->set_addr_align(0x8);
	symbolSection->set_entry_size(writer.get_default_entry_size(SHT_SYMTAB));

	ELFIO::string_section_accessor stringWriter(stringSection);
	ELFIO::symbol_section_accessor symbolWriter(writer, symbolSection);

	// Custom Nvidia info section with memory limits (regs, stack, frame) for each function

	auto infoSection = writer.sections.add(".nv.info");
	infoSection->set_type(SHT_LOPROC);
	infoSection->set_addr_align(0x4);
	infoSection->set_link(symbolSection->get_index());

	std::vector<char> infoBuffer;

	// Maintain a symbol offset for connecting the .text to the symbol (Nvidia requirement)

	auto symbolOffset = 0u;

	// Add functions .data and .text sections

	for (const auto& function : program->GetFunctions())
	{
		auto textSymbol = symbolOffset + 4;
		auto dataSymbol = symbolOffset + 2;

		// Add custom Nvidia info

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

		// Custom Nvidia info section for function properties (params, exits, s2rctaid)

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

		if (function->GetS2RCTAIDOffsetsCount() > 0)
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
			AppendBytes(functionInfoBuffer, DecomposeShort(function->GetS2RCTAIDOffsetsCount() * SZ_WORD)); // Size

			for (const auto& ctaOffset : function->GetS2RCTAIDOffsets())
			{
				AppendBytes(functionInfoBuffer, DecomposeWord(ctaOffset));
			}
		}

		if (function->GetExitOffsetsCount() > 0)
		{
			// EIATTR_EXIT_INSTR_OFFSETS
			//     Format: EIFMT_SVAL
			//     Value:  0x90
			//
			//     /*0000*/        .byte   0x04, 0x1c
			//     /*0002*/        .short  (.L_2 - .L_1)
			// .L_1:
			//     /*0004*/        .word   0x00000090
			// .L_2:

			AppendBytes(functionInfoBuffer, {(char)Type::EIFMT_SVAL});
			AppendBytes(functionInfoBuffer, {(char)Attribute::EIATTR_EXIT_INSTR_OFFSETS});
			AppendBytes(functionInfoBuffer, DecomposeShort(function->GetExitOffsetsCount() * SZ_WORD)); // Size

			for (const auto& exitOffset : function->GetExitOffsets())
			{
				AppendBytes(functionInfoBuffer, DecomposeWord(exitOffset));
			}
		}

		functionInfoSection->set_data(functionInfoBuffer.data(), functionInfoBuffer.size());

		// Add constant data section:
		// 	- 0x140 base
		// 	- Space for each parameter

		auto dataSize = ELF_SREG_SIZE + function->GetParametersSize();
		auto data = new char[dataSize](); // Zero initialized

		auto dataSection = writer.sections.add(".nv.constant0." + function->GetName());
		dataSection->set_type(SHT_PROGBITS);
		dataSection->set_flags(SHF_ALLOC);
		dataSection->set_addr_align(0x4);
		dataSection->set_data(data, dataSize);

		// Add .text section for the function body

		auto textSection = writer.sections.add(".text." + function->GetName());
		textSection->set_type(SHT_PROGBITS);
		textSection->set_flags(SHF_ALLOC | SHF_EXECINSTR);
		textSection->set_addr_align(0x20);
		textSection->set_link(symbolSection->get_index());
		textSection->set_info(function->GetRegisters() << 20 + textSymbol);
		textSection->set_data(function->GetText(), function->GetSize());

		// Link info/data sections to .text

		dataSection->set_info(textSection->get_index());
		functionInfoSection->set_info(textSection->get_index());

		// Update symbol table:
		//   - .text
		//   - .nv.constant0
		//   - _param
		//   - Global .text symbol

		symbolWriter.add_symbol(stringWriter,
			textSection->get_name().c_str(), 0x00000000, 0, STB_LOCAL, STT_SECTION, STV_DEFAULT, textSection->get_index()
		);
		symbolWriter.add_symbol(stringWriter,
			dataSection->get_name().c_str(), 0x00000000, 0, STB_LOCAL, STT_SECTION, STV_DEFAULT, dataSection->get_index()
		);
		symbolWriter.add_symbol(stringWriter,
			"_param", ELF_SREG_SIZE, function->GetParametersSize(), STB_LOCAL, STT_CUDA_OBJECT, STV_INTERNAL| STO_CUDA_CONSTANT, dataSection->get_index()
		);

		symbolWriter.add_symbol(stringWriter,
			function->GetName().c_str(), 0x00000000, textSection->get_size(), STB_GLOBAL, STT_FUNC, STV_DEFAULT | STO_CUDA_ENTRY, textSection->get_index()
		);

		symbolOffset += 4;
	}

	// Set full info, contains data for all functions

	infoSection->set_data(infoBuffer.data(), infoBuffer.size());

	// Output ELF binary to memory, copying data

	wmemstreambuf buffer(4096);
	std::ostream binaryStream(&buffer);
	writer.save(binaryStream);

	if (Utils::Options::Get<>(Utils::Options::Opt_Backend_dump))
	{
		writer.save("r3d3_dump.cubin");
	}

	std::size_t elfSize = 0;
	auto bufferContent = buffer.getcontent(elfSize);

	auto elfBinary = ::operator new(elfSize);
	std::memcpy(elfBinary, bufferContent, elfSize);

	Utils::Chrono::End(timeELF_start);

	if (Utils::Options::Get<>(Utils::Options::Opt_Print_debug))
	{
		Utils::Logger::LogDebug("Generate ELF binary, " + std::to_string(elfSize) + " bytes");
	}

	return new ELFBinary(elfBinary, elfSize);
}

}
