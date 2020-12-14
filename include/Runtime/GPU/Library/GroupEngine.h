#pragma once

#include <vector>

#include "HorseIR/Tree/Tree.h"

#include "Runtime/Runtime.h"
#include "Runtime/DataBuffers/DataBuffer.h"
#include "Runtime/DataBuffers/DictionaryBuffer.h"

namespace Runtime {
namespace GPU {

class GroupEngine
{
public:
	GroupEngine(Runtime& runtime, const HorseIR::Program *program) : m_runtime(runtime), m_program(program) {}

	DictionaryBuffer *Group(const std::vector<const DataBuffer *>& arguments);

private:
	const HorseIR::Function *GetFunction(const HorseIR::FunctionDeclaration *function) const;
	ListBuffer *ConstructListBuffer(TypedVectorBuffer<std::int64_t> *indexBuffer, TypedVectorBuffer<std::int64_t> *valuesBuffer) const;

	Runtime& m_runtime;
	const HorseIR::Program *m_program = nullptr;
};

}
}
