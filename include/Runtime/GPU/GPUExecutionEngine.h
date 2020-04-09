#pragma once

#include <vector>
#include <unordered_map>
#include <utility>

#include "Codegen/InputOptions.h"

#include "CUDA/Buffer.h"
#include "CUDA/KernelInvocation.h"

#include "PTX/FunctionOptions.h"

#include "Runtime/Runtime.h"
#include "Runtime/DataBuffers/DataBuffer.h"
#include "Runtime/DataBuffers/ListBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

#include "HorseIR/Tree/Tree.h"

namespace Runtime {

class GPUExecutionEngine
{
public:
	GPUExecutionEngine(Runtime& runtime, const HorseIR::Program *program) : m_runtime(runtime), m_program(program) {}

	std::vector<DataBuffer *> Execute(const HorseIR::Function *function, const std::vector<DataBuffer *>& arguments);

private:
	std::pair<unsigned int, unsigned int> GetBlockShape(Codegen::InputOptions *runtimeOptions, const PTX::FunctionOptions& kernelOptions) const;

	VectorBuffer *ResizeBuffer(const VectorBuffer *vectorBuffer, std::uint32_t size) const;
	ListBuffer *ResizeBuffer(const ListBuffer *listBuffer, const std::vector<std::uint32_t>& sizes) const;
	DataBuffer *ResizeBuffer(const DataBuffer *dataBuffer, CUDA::Buffer *sizeBuffer) const;

	template<typename T>
	void AllocateConstantParameter(CUDA::KernelInvocation& invocation, const T& value, const std::string& description) const;
	CUDA::Buffer *AllocateListSizeParameter(CUDA::KernelInvocation& invocation, const Analysis::ListShape *shape, const std::string& description) const;
	CUDA::Buffer *AllocateSizeParameter(CUDA::KernelInvocation& invocation, const Analysis::Shape *shape, bool returnParameter) const;

	Runtime& m_runtime;
	const HorseIR::Program *m_program = nullptr;

	// Cache for GPU execution

	std::unordered_map<const HorseIR::Function *, Codegen::InputOptions *> m_optionsCache;
};

}
