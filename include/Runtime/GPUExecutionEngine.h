#pragma once

#include <vector>
#include <utility>

#include "Codegen/InputOptions.h"
#include "Codegen/TargetOptions.h"

#include "CUDA/KernelInvocation.h"

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
	std::vector<std::uint32_t> GetCellSizes(const Analysis::ListShape *shape) const;

	std::pair<unsigned int, unsigned int> GetBlockShape(Codegen::InputOptions& runtimeOptions, const Codegen::TargetOptions& targetOptions, const PTX::FunctionOptions& kernelOptions) const;
	Codegen::InputOptions GetInputOptions(const HorseIR::Function *function, const std::vector<DataBuffer *>& arguments) const;

	void AllocateConstantParameter(CUDA::KernelInvocation& invocation, std::uint32_t value, const std::string& description) const;
	void AllocateCellSizes(CUDA::KernelInvocation& invocation, const Analysis::ListShape *shape, const std::string& description) const;
	void AllocateSizeBuffer(CUDA::KernelInvocation& invocation, const Analysis::Shape *shape) const;

	Runtime& m_runtime;
	const HorseIR::Program *m_program = nullptr;
};

}
