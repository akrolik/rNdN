#pragma once

#include <vector>

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
	GPUExecutionEngine(Runtime& runtime) : m_runtime(runtime) {}

	std::vector<DataBuffer *> Execute(const HorseIR::Function *function, const std::vector<DataBuffer *>& arguments);

private:
	unsigned int GetBlockSize(const Codegen::InputOptions& inputOptions, const Codegen::TargetOptions& targetOptions, const PTX::FunctionOptions& kernelOptions) const;

	Runtime& m_runtime;
};

}
