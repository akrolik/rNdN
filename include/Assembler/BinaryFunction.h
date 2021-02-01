#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

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

	std::size_t GetSharedMemorySize() const { return m_sharedMemorySize; }
	void SetSharedMemorySize(std::size_t size) { m_sharedMemorySize = size; }

private:
	std::string m_name;
	std::size_t m_registers = 0;
	std::size_t m_barriers = 0;

	std::vector<std::size_t> m_parameters;
	std::size_t m_parametersSize = 0;

	char *m_text = nullptr;
	std::size_t m_size = 0;

	std::vector<std::size_t> m_ctaOffsets;
	std::vector<std::size_t> m_exitOffsets;
	std::vector<std::size_t> m_coopOffsets;

	std::tuple<std::size_t, std::size_t, std::size_t> m_requiredThreads;
	std::tuple<std::size_t, std::size_t, std::size_t> m_maxThreads;

	bool m_ctaidzUsed = false;

	std::size_t m_sharedMemorySize = 0;
};

}
