#pragma once

#include <vector>

#include "Runtime/GPU/Library/LibraryEngine.h"

#include "Runtime/DataBuffers/DataBuffer.h"
#include "Runtime/DataBuffers/ListBuffer.h"

#include "HorseIR/Tree/Tree.h"

namespace Runtime {
namespace GPU {

class LoopJoinEngine : public LibraryEngine
{
public:
	using LibraryEngine::LibraryEngine;

	ListBuffer *Join(const std::vector<const DataBuffer *>& arguments);
};

}
}
