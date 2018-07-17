#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/Node.h"

#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/ModuleContent.h"

namespace HorseIR {

class Module : public Node
{
public:
	Module(std::string name, std::vector<ModuleContent *> contents) : m_name(name), m_contents(contents) {}

	std::string GetName() const { return m_name; }
	void SetName(std::string name) { m_name = name; }

	const std::vector<ModuleContent *>& GetContents() { return m_contents; }

	std::string ToString() const override
	{
		std::string code = "module " + m_name + " {\n";

		for (const auto& contents : m_contents)
		{
			code += "\t" + contents->ToString() + "\n";
		}

		return code + "}";
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

private:
	std::string m_name;
	std::vector<ModuleContent *> m_contents;
};

}
