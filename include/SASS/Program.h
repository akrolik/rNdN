#pragma once

#include <vector>

#include "SASS/Node.h"
#include "SASS/Function.h"

namespace SASS {

class Program : public Node
{
public:
	void AddFunction(Function *function) { m_functions.push_back(function); }
	const std::vector<Function *>& GetFunctions() const { return m_functions; }

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
	std::vector<Function *> m_functions;
};

};
