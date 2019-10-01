#pragma once

#include <vector>

#include "Runtime/DataObjects/DataObject.h"
#include "Runtime/Runtime.h"

#include "HorseIR/Tree/Tree.h"

namespace Runtime {

class BuiltinExecutionEngine
{
public:
	BuiltinExecutionEngine(Runtime& runtime) : m_runtime(runtime) {}

	std::vector<DataObject *> Execute(const HorseIR::BuiltinFunction *function, const std::vector<DataObject *>& arguments);

private:
	Runtime& m_runtime;
};

}
