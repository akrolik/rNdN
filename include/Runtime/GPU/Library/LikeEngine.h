#pragma once

#include <vector>

#include "Runtime/GPU/Library/LibraryEngine.h"

#include "Runtime/DataBuffers/DataBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

namespace Runtime {
namespace GPU {

class LikeEngine : public LibraryEngine
{
public:
	using LibraryEngine::LibraryEngine;

	TypedVectorBuffer<std::int8_t> *Like(const std::vector<const DataBuffer *>& arguments, bool cached = false);
};

}
}
