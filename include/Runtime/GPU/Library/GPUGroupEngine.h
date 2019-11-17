#pragma once

#include <utility>
#include <vector>

#include "Codegen/InputOptions.h"

#include "Runtime/Runtime.h"
#include "Runtime/DataBuffers/DictionaryBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

#include "HorseIR/Tree/Tree.h"

namespace Runtime {

class GPUGroupEngine
{
public:
	GPUGroupEngine(Runtime& runtime) : m_runtime(runtime) {}

	DictionaryBuffer *Group(const std::vector<VectorBuffer *>& dataBuffers);

private:
	Codegen::InputOptions GenerateInputOptions(const Analysis::VectorShape *vectorShape, const HorseIR::Function *groupFunction) const;

	std::pair<HorseIR::Program *, HorseIR::Function *> GenerateProgram(const std::vector<VectorBuffer *>& columns) const;
	HorseIR::Function *GenerateGroupFunction(const std::vector<const HorseIR::BasicType *>& types) const;

	Runtime& m_runtime;
};

}
