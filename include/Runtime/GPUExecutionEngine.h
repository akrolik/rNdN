#pragma once

#include <vector>

#include "Codegen/InputOptions.h"
#include "Codegen/TargetOptions.h"

#include "PTX/FunctionOptions.h"

#include "Runtime/DataObjects/DataObject.h"
#include "Runtime/Runtime.h"

#include "HorseIR/Tree/Tree.h"

namespace Runtime {

class GPUExecutionEngine
{
public:
	GPUExecutionEngine(Runtime& runtime) : m_runtime(runtime) {}

	std::vector<DataObject *> Execute(const HorseIR::Function *function, const std::vector<DataObject *>& arguments);

private:
	unsigned int GetBlockSize(const Codegen::InputOptions& inputOptions, const Codegen::TargetOptions& targetOptions, const PTX::FunctionOptions& kernelOptions) const;

	Runtime& m_runtime;
};

}
