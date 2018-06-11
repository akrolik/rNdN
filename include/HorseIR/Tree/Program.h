#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/Node.h"

#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Module.h"

namespace HorseIR {

class Program : public Node
{
public:
	Program(std::vector<Module *> modules) : m_modules(modules) {}

	const std::vector<Module *>& GetModules() const { return m_modules; }

	std::string ToString() const override
	{
		std::string code;
		for (auto module : m_modules)
		{
			code += module->ToString() + "\n";
		}
		return code;
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

private:
	std::vector<Module *> m_modules;
};

}
