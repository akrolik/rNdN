#pragma once

#include <vector>

#include "Runtime/GPU/Library/LibraryEngine.h"

#include "Runtime/DataBuffers/DataBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

#include "HorseIR/Tree/Tree.h"

namespace Runtime {
namespace GPU {

class UniqueEngine : public LibraryEngine
{
public:
	using LibraryEngine::LibraryEngine;

	TypedVectorBuffer<std::int64_t> *Unique(const std::vector<const DataBuffer *>& arguments);
};

}
}
