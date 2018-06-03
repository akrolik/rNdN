#pragma once

#include <string>

#include "PTX/Module.h"

namespace PTX {

class Program
{
public:
	void AddModule(Module *module) { m_modules.push_back(module); }

	std::string ToString() const
	{
		std::ostringstream code;

		for (auto it = m_modules.cbegin(); it != m_modules.cend(); ++it)
		{
			code << (*it)->ToString();
		}

		return code.str();
	}

private:
	std::vector<Module *> m_modules;
};

}
