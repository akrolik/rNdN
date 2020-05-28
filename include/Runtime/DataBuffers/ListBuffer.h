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

	virtual ~ListBuffer() override;
	virtual ListBuffer *Clone() const override = 0;

	// Type/Shape

	const HorseIR::ListType *GetType() const override { return m_type; }
	const Analysis::ListShape *GetShape() const override { return m_shape; }

	// Cells

	virtual const std::vector<DataBuffer *>& GetCells() const = 0;
	virtual DataBuffer *GetCell(unsigned int index) const = 0;
	virtual size_t GetCellCount() const = 0;

	virtual void ResizeCells(unsigned int size) = 0;

protected:
	ListBuffer(HorseIR::ListType *type, Analysis::ListShape *shape);

	HorseIR::ListType *m_type = nullptr;
	Analysis::ListShape *m_shape = nullptr;
};

}
