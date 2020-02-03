#pragma once

#include <utility>
#include <vector>

#include "Runtime/Runtime.h"
#include "Runtime/DataBuffers/DataBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

namespace Runtime {

class GPUSortEngine
{
public:
	GPUSortEngine(Runtime& runtime) : m_runtime(runtime) {}

	std::pair<TypedVectorBuffer<std::int64_t> *, DataBuffer *> Sort(const std::vector<DataBuffer *>& arguments);

private:
	Runtime& m_runtime;
};

}
