#pragma once

#include "Runtime/DataBuffers/DataBuffer.h"

#include <string>
#include <vector>

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Tree/Tree.h"

namespace Runtime {

class ListBuffer : public DataBuffer
{
public:
	constexpr static DataBuffer::Kind BufferKind = DataBuffer::Kind::List;

	virtual ~ListBuffer() override;
	virtual ListBuffer *Clone() const override = 0;

	// Type/Shape

	const HorseIR::ListType *GetType() const override { return m_type; }
	const HorseIR::Analysis::ListShape *GetShape() const override { return m_shape; }

	// Cells

	virtual std::vector<const DataBuffer *> GetCells() const = 0;
	virtual std::vector<DataBuffer *>& GetCells() = 0;

	virtual const DataBuffer *GetCell(unsigned int index) const = 0;
	virtual DataBuffer *GetCell(unsigned int index) = 0;

	virtual size_t GetCellCount() const = 0;

	// Sizing

	virtual void ResizeCells(unsigned int size) = 0;

protected:
	ListBuffer(HorseIR::ListType *type, HorseIR::Analysis::ListShape *shape);
	ListBuffer() : ListBuffer(nullptr, nullptr) {}

	// Type/shape

	HorseIR::ListType *m_type = nullptr;
	HorseIR::Analysis::ListShape *m_shape = nullptr;
};

}
