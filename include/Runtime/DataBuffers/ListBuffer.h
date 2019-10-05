#pragma once

#include "Runtime/DataBuffers/DataBuffer.h"

#include <string>
#include <vector>

#include "HorseIR/Tree/Tree.h"

namespace Runtime {

class ListBuffer : public DataBuffer
{
public:
	ListBuffer(DataBuffer *cell) : ListBuffer(std::vector<DataBuffer *>({cell})) {}
	ListBuffer(const std::vector<DataBuffer *>& cells) : m_cells(cells)
	{
		std::vector<HorseIR::Type *> cellTypes;
		for (const auto& cell : cells)
		{
			cellTypes.push_back(cell->GetType()->Clone());
		}
		m_type = new HorseIR::ListType(cellTypes);
	}

	const HorseIR::ListType *GetType() const { return m_type; }

	// Cells

	void AddCell(DataBuffer *cell);
	DataBuffer *GetCell(unsigned int index) { return m_cells.at(index); }
	size_t GetCellCount() const { return m_cells.size(); }

	// CPU/GPU management

	CUDA::Buffer *GetGPUWriteBuffer() override { Utils::Logger::LogError("Unable to allocate list GPU buffer"); }
	CUDA::Buffer *GetGPUReadBuffer() const override { Utils::Logger::LogError("Unable to allocate list GPU buffer"); }

	// Printers

	std::string Description() const override;
	std::string DebugDump() const override;

private:
	HorseIR::ListType *m_type = nullptr;

	std::vector<DataBuffer *> m_cells;
};

}
