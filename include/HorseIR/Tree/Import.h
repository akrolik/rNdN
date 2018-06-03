#pragma once

#include <string>

#include "HorseIR/Tree/ModuleContent.h"

namespace HorseIR {

class Import : public ModuleContent
{
public:
	Import(std::string name) : m_name(name) {}

	std::string ToString() const
	{
		return "import " + m_name + ";";
	}

private:
	std::string m_name;
};

}
