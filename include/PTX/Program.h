#pragma once

#include <string>

#include "PTX/Module.h"

namespace PTX {

class Program
{
public:
	void AddModule(Module *module) { m_modules.push_back(module); }

	const std::vector<Module *>& GetModules() const { return m_modules; }

private:
	std::vector<Module *> m_modules;
};

}
