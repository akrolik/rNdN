#pragma once

#include <vector>
#include <utility>

#include "Frontend/Codegen/InputOptions.h"

#include "CUDA/Constant.h"
#include "CUDA/KernelInvocation.h"
#include "CUDA/Kernel.h"

#include "PTX/Tree/Tree.h"

#include "Runtime/Runtime.h"
#include "Runtime/DataBuffers/DataBuffer.h"
#include "Runtime/DataBuffers/ListBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

#include "HorseIR/Tree/Tree.h"

#include "Libraries/robin_hood.h"

namespace Runtime {
namespace GPU {

class ExecutionEngine
{
public:
	ExecutionEngine(Runtime& runtime, const HorseIR::Program *program) : m_runtime(runtime), m_program(program) {}

	std::vector<DataBuffer *> Execute(const HorseIR::Function *function, const std::vector<const DataBuffer *>& arguments);

private:
	std::pair<unsigned int, unsigned int> GetBlockShape(Frontend::Codegen::InputOptions *runtimeOptions, const PTX::FunctionDefinition<PTX::VoidType> * kernelCode, const CUDA::Kernel& kernel) const;

	template<typename T>
	CUDA::TypedConstant<T> *AllocateConstantParameter(CUDA::KernelInvocation& invocation, const T& value, const std::string& description) const;
	template<typename T>
	CUDA::ConstantBuffer *AllocateConstantVectorParameter(CUDA::KernelInvocation &invocation, const CUDA::Vector<T>& values, const std::string& description) const;

	Runtime& m_runtime;
	const HorseIR::Program *m_program = nullptr;

	// Cache for GPU execution

	robin_hood::unordered_map<const HorseIR::Function *, Frontend::Codegen::InputOptions *> m_optionsCache;
};

}
}
