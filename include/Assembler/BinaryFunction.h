#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "SASS/Tree/Tree.h"
#include "SASS/Utils/PrettyPrinter.h"

#include "Utils/Format.h"

namespace Assembler {

class BinaryFunction
{
public:
	BinaryFunction(const std::string& name) : m_name(name) {}

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	// Number of registers in text

	void SetRegisters(const std::size_t registers) { m_registers = registers; }
	std::size_t GetRegisters() const { return m_registers; }

	// Number of barriers (high water mark)

	std::size_t GetBarriers() const { return m_barriers; }
	void SetBarriers(std::size_t barriers) { m_barriers = barriers; }

	// Parameters (specified by size)

	const std::vector<std::size_t>& GetParameters() const { return m_parameters; }

	std::size_t GetParametersCount() const { return m_parameters.size(); }
	std::size_t GetParametersSize() const { return m_parametersSize; }

	void AddParameter(std::size_t parameter)
	{
		m_parameters.push_back(parameter);
		m_parametersSize += parameter;
	}
	void SetParameters(const std::vector<std::size_t>& parameters) { m_parameters = parameters; }

	// Binary text

	const char *GetText() const { return m_text; }
	std::size_t GetSize() const { return m_size; }

	void SetText(char *text, std::size_t size)
	{
		m_text = text;
		m_size = size;
	}

	const std::vector<const SASS::Instruction *>& GetLinearProgram() const { return m_linearProgram; }
	void SetLinearProgram(const std::vector<const SASS::Instruction *>& linearProgram) { m_linearProgram = linearProgram; }
	
	// S2R CTAID instruction offsets

	const std::vector<std::size_t>& GetS2RCTAIDOffsets() const { return m_ctaOffsets; }
	std::size_t GetS2RCTAIDOffsetsCount() const { return m_ctaOffsets.size(); }

	void AddS2RCTAIDOffset(std::size_t ctaOffset) { m_ctaOffsets.push_back(ctaOffset); }
	void SetS2RCTAIDOffsets(const std::vector<std::size_t>& ctaOffsets) { m_ctaOffsets = ctaOffsets; }

	// Exit offsets

	const std::vector<std::size_t>& GetExitOffsets() const { return m_exitOffsets; }
	std::size_t GetExitOffsetsCount() const { return m_exitOffsets.size(); }

	void AddExitOffset(std::size_t exitOffset) { m_exitOffsets.push_back(exitOffset); }
	void SetExitOffsets(const std::vector<std::size_t>& exitOffsets) { m_exitOffsets = exitOffsets; }

	// Coop offsets

	const std::vector<std::size_t>& GetCoopOffsets() const { return m_coopOffsets; }
	std::size_t GetCoopOffsetsCount() const { return m_coopOffsets.size(); }

	void AddCoopOffset(std::size_t coopOffset) { m_coopOffsets.push_back(coopOffset); }
	void SetCoopOffsets(const std::vector<std::size_t>& coopOffsets) { m_coopOffsets = coopOffsets; }

	// Thread Properties

	std::tuple<std::size_t, std::size_t, std::size_t> GetRequiredThreads() const { return m_requiredThreads; }
	void SetRequiredThreads(std::size_t dimX, std::size_t dimY = 1, std::size_t dimZ = 1) { m_requiredThreads = { dimX, dimY, dimZ }; }

	std::tuple<std::size_t, std::size_t, std::size_t> GetMaxThreads() const { return m_maxThreads; }
	void SetMaxThreads(std::size_t dimX, std::size_t dimY = 1, std::size_t dimZ = 1) { m_maxThreads = { dimX, dimY, dimZ }; }

	// CTAID Z Dimension

	bool GetCTAIDZUsed() const { return m_ctaidzUsed; }
	void SetCTAIDZUsed(bool ctaidzUsed) { m_ctaidzUsed = ctaidzUsed; }

	// Shared Memory

	struct Variable {
		std::string Name;
		std::size_t Size;
		std::size_t DataSize;
	};

	const std::vector<Variable>& GetSharedVariables() const { return m_sharedVariables; }
	std::size_t GetSharedVariableCount() const { return m_sharedVariables.size(); }

	void AddSharedVariable(const std::string& name, std::size_t size, std::size_t dataSize)
	{
		m_sharedVariables.push_back({ name, size, dataSize });
	}
	void SetSharedVariables(const std::vector<Variable>& sharedVariables) { m_sharedVariables = sharedVariables; }

	// Constant Memory

	std::size_t GetConstantMemorySize() const { return m_constantMemory.size(); }
	const std::vector<char>& GetConstantMemory() const { return m_constantMemory; }

	void SetConstantMemory(const std::vector<char>& constantMemory) { m_constantMemory = constantMemory; }

	// Relocations

	enum class RelocationKind {
		ABS32_LO_20,
		ABS32_HI_20,
		ABS24_20
	};

	static std::string RelocationKindString(RelocationKind kind)
	{
		switch (kind)
		{
			case RelocationKind::ABS32_LO_20:
				return "ABS32_LO_20";
			case RelocationKind::ABS32_HI_20:
				return "ABS32_HI_20";
			case RelocationKind::ABS24_20:
				return "ABS24_20";
		}
		return "<unknown>";
	}

	struct Relocation {
		std::string Name;
		std::size_t Address;
		RelocationKind Kind;
	};
	
	const std::vector<Relocation>& GetRelocations() const { return m_relocations; }
	std::size_t GetRelocationsCount() const { return m_relocations.size(); }

	void AddRelocation(const std::string& name, std::size_t address, RelocationKind kind)
	{
		m_relocations.push_back({ name, address, kind });
	}
	void SetRelocations(const std::vector<Relocation>& relocations) { m_relocations = relocations; }

	// Indirect branches

	struct IndirectBranch {
		std::size_t Offset;
		std::size_t Target;
	};

	const std::vector<IndirectBranch>& GetIndirectBranches() const { return m_indirectBranches; }
	std::size_t GetIndirectBranchesCount() const { return m_indirectBranches.size(); }

	void AddIndirectBranch(std::size_t offset, std::size_t target)
	{
		m_indirectBranches.push_back({ offset, target });
	}
	void SetIndirectBranches(const std::vector<IndirectBranch>& indirectBranches) { m_indirectBranches = indirectBranches; }

	// Stack size

	std::size_t GetCRSStackSize() const { return m_crsStackSize; }
	void SetCRSStackSize(std::size_t crsStackSize) { m_crsStackSize = crsStackSize; }

	// Formatting

	std::string ToString() const
	{
		std::string code;
		code += "// Binary SASS Function: " + m_name + "\n";

		// Metadata memory formatting

		if (m_parameters.size() > 0)
		{
			code += "// - Parameters (bytes): ";
			auto first = true;
			for (const auto parameter : m_parameters)
			{
				if (!first)
				{
					code += ", ";
				}
				first = false;
				code += std::to_string(parameter);
			}
			code += "\n";
		}
		code += "// - Registers: " + std::to_string(m_registers) + "\n";
		code += "// - Barriers: " + std::to_string(m_barriers) + "\n";

		// Metadata offsets formatting

		if (m_ctaOffsets.size() > 0)
		{
			code += "// - S2RCTAID Offsets: ";
			auto first = true;
			for (const auto offset : m_ctaOffsets)
			{
				if (!first)
				{
					code += ", ";
				}
				first = false;
				code += Utils::Format::HexString(offset, 4);
			}
			code += "\n";
		}
		if (m_exitOffsets.size() > 0)
		{
			code += "// - Exit Offsets: ";
			auto first = true;
			for (const auto offset : m_exitOffsets)
			{
				if (!first)
				{
					code += ", ";
				}
				first = false;
				code += Utils::Format::HexString(offset, 4);
			}
			code += "\n";
		}
		if (m_coopOffsets.size() > 0)
		{
			code += "// - Coop Offsets: ";
			auto first = true;
			for (const auto offset : m_coopOffsets)
			{
				if (!first)
				{
					code += ", ";
				}
				first = false;
				code += Utils::Format::HexString(offset, 4);
			}
			code += "\n";
		}

		// Thread metadata

		if (auto [dimX, dimY, dimZ] = m_requiredThreads; dimX > 0)
		{
			code += "// - Required Threads: ";
			code += std::to_string(dimX) + ", ";
			code += std::to_string(dimY) + ", ";
			code += std::to_string(dimZ);
			code += "\n";
		}
		else if (auto [dimX, dimY, dimZ] = m_maxThreads; dimX > 0)
		{
			code += "// - Max Threads: ";
			code += std::to_string(dimX) + ", ";
			code += std::to_string(dimY) + ", ";
			code += std::to_string(dimZ);
			code += "\n";
		}

		if (m_ctaidzUsed)
		{
			code += "// - CTAIDZ\n";
		}

		// Shared memory

		if (m_sharedVariables.size() > 0)
		{
			code += "// - Shared Memory:\n";
			for (const auto& variable : m_sharedVariables)
			{
				code += ".shared " + variable.Name + " { ";
				code += "size=" + Utils::Format::HexString(variable.Size) + " bytes; ";
				code += "datasize=" + Utils::Format::HexString(variable.DataSize) + " bytes }\n";
			}
		}

		// Constant memory

		if (m_constantMemory.size() > 0)
		{
			code += "// - Constant Memory: " + std::to_string(m_constantMemory.size()) + " bytes\n";
		}

		// Relocations

		if (m_relocations.size() > 0)
		{
			code += "// - Relocations:\n";
			for (const auto& relocation : m_relocations)
			{
				code += ".reloc " + relocation.Name + " " + RelocationKindString(relocation.Kind) + " (" + Utils::Format::HexString(relocation.Address) + ")\n";
			}
		}

		// Indirect branches

		if (m_indirectBranches.size() > 0)
		{
			code += "// - Indirect Branches:\n";
			for (const auto& branch : m_indirectBranches)
			{
				code += ".branch " + Utils::Format::HexString(branch.Offset) + " -> " + Utils::Format::HexString(branch.Target) + "\n";
			}
		}

		// Stack size

		if (m_crsStackSize > 0)
		{
			code += "// - CRS Stack Size: " + Utils::Format::HexString(m_crsStackSize) + " bytes \n";
		}

		// Print assembled program with address and binary format

		auto first = true;
		for (auto i = 0u; i < m_linearProgram.size(); ++i)
		{
			auto instruction = m_linearProgram.at(i);

			auto address = "/* " + Utils::Format::HexString(i * sizeof(std::uint64_t), 4) + " */    ";
			auto mnemonic = SASS::PrettyPrinter::PrettyString(instruction);
			auto binary = "/* " + Utils::Format::HexString(instruction->ToBinary(), 16) + " */";

			auto indent = 4;
			auto length = mnemonic.length();
			if (length < 48)
			{
				indent = 48 - length;
			}
			std::string spacing(indent, ' ');

			if (!first)
			{
				code += "\n";
			}
			first = false;
			code += address + mnemonic + spacing + binary;
		}
		return code;
	}

private:
	std::string m_name;
	std::size_t m_registers = 0;
	std::size_t m_barriers = 0;

	std::vector<std::size_t> m_parameters;
	std::size_t m_parametersSize = 0;
	std::size_t m_crsStackSize = 0;

	std::vector<const SASS::Instruction *> m_linearProgram;
	char *m_text = nullptr;
	std::size_t m_size = 0;

	std::vector<std::size_t> m_ctaOffsets;
	std::vector<std::size_t> m_exitOffsets;
	std::vector<std::size_t> m_coopOffsets;

	std::tuple<std::size_t, std::size_t, std::size_t> m_requiredThreads;
	std::tuple<std::size_t, std::size_t, std::size_t> m_maxThreads;

	bool m_ctaidzUsed = false;

	std::vector<Variable> m_sharedVariables;
	std::vector<char> m_constantMemory;

	std::vector<Relocation> m_relocations;
	std::vector<IndirectBranch> m_indirectBranches;
};

}
