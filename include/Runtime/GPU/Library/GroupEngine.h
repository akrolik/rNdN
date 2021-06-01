#pragma once

#include <vector>

#include "Runtime/GPU/Library/LibraryEngine.h"

#include "Runtime/DataBuffers/DataBuffer.h"
#include "Runtime/DataBuffers/DictionaryBuffer.h"

#include "HorseIR/Tree/Tree.h"

namespace Runtime {
namespace GPU {

class GroupEngine : public LibraryEngine
{
public:
	using LibraryEngine::LibraryEngine;

	DictionaryBuffer *Group(const std::vector<const DataBuffer *>& arguments);

private:
	ListBuffer *ConstructListBuffer(TypedVectorBuffer<std::int64_t> *indexBuffer, TypedVectorBuffer<std::int64_t> *valuesBuffer) const;
};

}
}
