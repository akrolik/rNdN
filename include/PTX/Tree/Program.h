#pragma once

#include "PTX/Tree/Node.h"

#include <vector>

#include "PTX/Tree/Module.h"
#include "PTX/Tree/Functions/Function.h"

#include "Utils/Logger.h"

namespace PTX {

class Program : public Node
{
public:
	void AddModule(const Module *module) { m_modules.push_back(module); }

	const std::vector<const Module *>& GetModules() const { return m_modules; }

	const Function *GetEntryFunction(const std::string& name) const
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

	// Visitors

	void Accept(ConstHierarchicalVisitor& visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			for (const auto& module : m_modules)
			{
				module->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

private:
	std::vector<const Module *> m_modules;
};

}
