#pragma once

#include "PTX/Node.h"

#include <vector>

#include "PTX/Module.h"
#include "PTX/Functions/Function.h"

#include "Utils/Logger.h"

namespace PTX {

class Program : public Node
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

	std::string ToString(unsigned int indentation) const override
	{
		std::string code = "";

		bool first = true;
		for (const auto& module : m_modules)
		{
			if (!first)
			{
				code += "\n\n";
			}
			first = false;
			code += module->ToString(0);
		}

		return code;
	}

	json ToJSON() const override
	{
		json j;
		for (const auto& module : m_modules)
		{
			j["modules"].push_back(module->ToJSON());
		}
		return j;
	}

private:
	std::vector<const Module *> m_modules;
};

}
