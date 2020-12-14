#pragma once

#include <vector>

#include "SASS/Node.h"
#include "SASS/Function.h"

namespace SASS {

class Program : public Node
{
public:
	// Compute capability

	unsigned int GetComputeCapability() const { return m_computeCapability; }
	void SetComputeCapability(unsigned int computeCapability) { m_computeCapability = computeCapability; }

	// Functions

	std::vector<const Function *> GetFunctions() const
	{
		return { std::begin(m_functions), std::end(m_functions) };
	}
	std::vector<Function *>& GetFunctions() { return m_functions; }

	void AddFunction(Function *function) { m_functions.push_back(function); }
	void SetFunctions(const std::vector<Function *>& functions) { m_functions = functions; }

	// Formatting

	std::string ToString() const override
	{
		std::string code;

		auto first = true;
		for (const auto& function : m_functions)
		{
			if (!first)
			{
				code += "\n\n";
			}
			first = false;
			code += function->ToString();
		}

		return code;
	}

private:
	unsigned int m_computeCapability = 0;
	std::vector<Function *> m_functions;
};

};
