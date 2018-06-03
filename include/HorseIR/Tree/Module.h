#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/ModuleContent.h"
#include "HorseIR/Tree/Node.h"

namespace HorseIR {

class Module : public Node
{
public:
	Module(std::string name, std::vector<ModuleContent *> contents) : m_name(name), m_contents(contents) {}

	std::string GetName() const { return m_name; }
	void SetName(std::string name) { m_name = name; }

	std::string ToString() const
	{
		std::string code = "module " + m_name + " {\n";

		for (auto it = m_contents.cbegin(); it != m_contents.cend(); ++it)
		{
			code += "\t" + (*it)->ToString() + "\n";
		}

		return code + "}";
	}

private:
	std::string m_name;
	std::vector<ModuleContent *> m_contents;
};

}
