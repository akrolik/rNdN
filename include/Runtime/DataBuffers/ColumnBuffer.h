#pragma once

#include <string>

#include "Runtime/DataBuffers/DataBuffer.h"

namespace Runtime {

class ColumnBuffer : public DataBuffer
{
public:
	~ColumnBuffer() override;

	virtual ColumnBuffer *Clone() const = 0;

	// Column size

	virtual unsigned int GetElementCount() const = 0;

	// Colum printing

	virtual std::string DebugDump() const = 0;
	virtual std::string DebugDump(unsigned int index) const = 0;

protected:
	ColumnBuffer(DataBuffer::Kind kind) : DataBuffer(kind) {}
};

}
