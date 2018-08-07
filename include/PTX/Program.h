#pragma once

#include <vector>

#include "PTX/Module.h"
#include "PTX/Functions/Function.h"

#include "Utils/Logger.h"

namespace PTX {

class Program
{
public:
	void AddModule(const Module *module) { m_modules.push_back(module); }

	const std::vector<const Module *>& GetModules() const { return m_modules; }

	const PTX::Function *GetEntryFunction(const std::string& name) const
	{
		for (const auto& module : m_modules)
		{
			if (module->ContainsEntryFunction(name))
			{
				return module->GetEntryFunction(name);
			}
		}
		Utils::Logger::LogError("Cannot find entry fuction '" + name);
	}

private:
	std::vector<const Module *> m_modules;
};

}
