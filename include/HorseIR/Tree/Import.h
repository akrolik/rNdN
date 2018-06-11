#pragma once

#include <string>

#include "HorseIR/Tree/ModuleContent.h"

#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class Import : public ModuleContent
{
public:
	Import(std::string name) : m_name(name) {}

	std::string GetName() const { return m_name; }
	void SetName(std::string name) { m_name = name; }

	std::string ToString() const override
	{
		return "import " + m_name + ";";
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

private:
	std::string m_name;
};

}
