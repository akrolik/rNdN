#pragma once

#include <vector>

#include "Runtime/Runtime.h"
#include "Runtime/DataBuffers/DataBuffer.h"
#include "Runtime/DataBuffers/DictionaryBuffer.h"

namespace Runtime {

class GPUGroupEngine
{
public:
	GPUGroupEngine(Runtime& runtime) : m_runtime(runtime) {}

	DictionaryBuffer *Group(const std::vector<DataBuffer *>& arguments);

private:
	Runtime& m_runtime;
};

}
