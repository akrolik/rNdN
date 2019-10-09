#pragma once

#include "Runtime/DataBuffers/DataBuffer.h"

#include <string>
#include <vector>

#include "HorseIR/Tree/Tree.h"

namespace Runtime {

class ListBuffer : public DataBuffer
{
public:
	constexpr static DataBuffer::Kind BufferKind = DataBuffer::Kind::List;

	ListBuffer(DataBuffer *cell) : ListBuffer(std::vector<DataBuffer *>({cell})) {}
	ListBuffer(const std::vector<DataBuffer *>& cells) : DataBuffer(DataBuffer::Kind::List), m_cells(cells)
	{
		std::vector<HorseIR::Type *> cellTypes;
		std::vector<const Analysis::Shape *> cellShapes;
		for (const auto& cell : cells)
		{
			cellTypes.push_back(cell->GetType()->Clone());
			cellShapes.push_back(cell->GetShape());
		}
		m_type = new HorseIR::ListType(cellTypes);
		m_shape = new Analysis::ListShape(new Analysis::Shape::ConstantSize(cells.size()), cellShapes);
	}

	const HorseIR::ListType *GetType() const override { return m_type; }
	const Analysis::ListShape *GetShape() const override { return m_shape; }

	// Cells

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
	Analysis::ListShape *m_shape = nullptr;

	std::vector<DataBuffer *> m_cells;
};

}
