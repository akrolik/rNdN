#pragma once

#include <utility>
#include <vector>

#include "Runtime/GPU/Library/LibraryEngine.h"

#include "Runtime/DataBuffers/DataBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

#include "HorseIR/Tree/Tree.h"

namespace Runtime {
namespace GPU {

class SortEngine : public LibraryEngine
{
public:
	using LibraryEngine::LibraryEngine;

	std::pair<TypedVectorBuffer<std::int64_t> *, DataBuffer *> Sort(const std::vector<const DataBuffer *>& arguments);
};

}
}
