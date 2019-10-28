#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include "Codegen/InputOptions.h"

#include "Runtime/Runtime.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

#include "HorseIR/Tree/Tree.h"

namespace Runtime {

class GPUSortEngine
{
public:
	GPUSortEngine(Runtime& runtime) : m_runtime(runtime) {}

	VectorBuffer *Sort(const std::vector<VectorBuffer *>& columns, const std::vector<char>& orders);

private:
	std::pair<Codegen::InputOptions, Codegen::InputOptions> GenerateInputOptions(
		const Analysis::VectorShape *vectorShape, const std::vector<const Analysis::VectorShape *>& dataShapes,
		const HorseIR::Function *initFunction, const HorseIR::Function *sortFunction
	) const;

	std::tuple<HorseIR::Program *, HorseIR::Function *, HorseIR::Function *> GenerateProgram(const std::vector<VectorBuffer *>& columns, const std::vector<char>& orders) const;
	HorseIR::Function *GenerateInitFunction(const std::vector<const HorseIR::BasicType *>& types, const std::vector<char>& orders) const;
	HorseIR::Function *GenerateSortFunction(const std::vector<const HorseIR::BasicType *>& types, const std::vector<char>& orders) const;

	Runtime& m_runtime;
};

}
