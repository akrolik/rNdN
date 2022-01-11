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

	// Column printing

	virtual std::string DebugDumpElement(unsigned int index, unsigned int indent = 0, bool preindent = false) const = 0;

protected:
	ColumnBuffer(DataBuffer::Kind kind) : DataBuffer(kind) {}
};

}
