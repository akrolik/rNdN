#pragma once

#include <vector>
#include <utility>

#include "Codegen/InputOptions.h"
#include "Codegen/TargetOptions.h"

#include "PTX/FunctionOptions.h"

#include "Runtime/Runtime.h"
#include "Runtime/DataBuffers/DataBuffer.h"

#include "HorseIR/Tree/Tree.h"

namespace Runtime {

class GPUExecutionEngine
{
public:
	GPUExecutionEngine(Runtime& runtime, const HorseIR::Program *program) : m_runtime(runtime), m_program(program) {}

	std::vector<DataBuffer *> Execute(const HorseIR::Function *function, const std::vector<DataBuffer *>& arguments);

private:
	std::pair<unsigned int, unsigned int> GetBlockShape(const Codegen::InputOptions& inputOptions, const Codegen::TargetOptions& targetOptions, const PTX::FunctionOptions& kernelOptions) const;

	Runtime& m_runtime;
	const HorseIR::Program *m_program = nullptr;
};

}
