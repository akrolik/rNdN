#pragma once

#include <vector>

#include "Assembler/BinaryFunction.h"

namespace Assembler {

class BinaryProgram
{
public:
	void AddFunction(BinaryFunction *function) { m_functions.push_back(function); }
	const std::vector<BinaryFunction *>& GetFunctions() const { return m_functions; }
	unsigned int GetFunctionCount() const { return m_functions.size(); }

	unsigned int GetComputeCapability() const { return m_computeCapability; }
	void SetComputeCapability(unsigned int computeCapability) { m_computeCapability = computeCapability; }

private:
	std::vector<BinaryFunction *> m_functions;

	unsigned int m_computeCapability = 0;
};

}
