#pragma once

#include <vector>

#include "HorseIR/Tree/Node.h"

#include "HorseIR/Tree/Module.h"

namespace HorseIR {

class Program : public Node
{
public:
	Program(std::vector<Module *> modules) : m_modules(modules) {}

	const std::vector<Module *>& GetModules() const { return m_modules; }

	std::string ToString() const
	{
		std::string code;
		for (auto it = m_modules.cbegin(); it != m_modules.cend(); ++it)
		{
			code += (*it)->ToString() + "\n";
		}
		return code;
	}

private:
	std::vector<Module *> m_modules;
};

}
