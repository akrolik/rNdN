#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace Assembler {

class BinaryFunction
{
public:
	BinaryFunction(const std::string& name) : m_name(name) {}

	const std::string& GetName() const { return m_name; }

	void SetRegisters(const std::size_t registers) { m_registers = registers; }
	std::size_t GetRegisters() const { return m_registers; }

	void AddParameter(std::size_t parameter)
	{
		m_parameters.push_back(parameter);
		m_parametersSize += parameter;
	}
	const std::vector<std::size_t>& GetParameters() const { return m_parameters; }
	std::size_t GetParametersCount() const { return m_parameters.size(); }
	std::size_t GetParametersSize() const { return m_parametersSize; }

	void SetText(char *text, std::size_t size)
	{
		m_text = text;
		m_size = size;
	}
	char *GetText() const { return m_text; }
	std::size_t GetSize() const { return m_size; }

	void SetS2RCTAIDOffsets(const std::vector<std::size_t>& ctaOffsets) { m_ctaOffsets = ctaOffsets; }
	const std::vector<std::size_t>& GetS2RCTAIDOffsets() const { return m_ctaOffsets; }
	std::size_t GetS2RCTAIDOffsetsCount() const { return m_ctaOffsets.size(); }

	void SetExitOffsets(const std::vector<std::size_t>& exitOffsets) { m_exitOffsets = exitOffsets; }
	const std::vector<std::size_t>& GetExitOffsets() const { return m_exitOffsets; }
	std::size_t GetExitOffsetsCount() const { return m_exitOffsets.size(); }

private:
	std::string m_name;
	std::size_t m_registers = 0;

	std::vector<std::size_t> m_parameters;
	std::size_t m_parametersSize = 0;

	char *m_text = nullptr;
	std::size_t m_size = 0;

	std::vector<std::size_t> m_ctaOffsets;
	std::vector<std::size_t> m_exitOffsets;
};

}