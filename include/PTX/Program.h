#pragma once

#include <vector>

#include "PTX/Module.h"

namespace PTX {

class Program
{
public:
	void AddModule(const Module *module) { m_modules.push_back(module); }

	const std::vector<const Module *>& GetModules() const { return m_modules; }

private:
	std::vector<const Module *> m_modules;
};

}
