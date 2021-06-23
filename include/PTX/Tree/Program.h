#pragma once

#include "PTX/Tree/Node.h"

#include <vector>

#include "PTX/Tree/Module.h"
#include "PTX/Tree/Functions/FunctionDefinition.h"

#include "Utils/Logger.h"

namespace PTX {

class Program : public Node
{
public:
	// Properties

	std::vector<const Module *> GetModules() const
	{
		return { std::begin(m_modules), std::end(m_modules) };
	}
	std::vector<Module *>& GetModules() { return m_modules; }
	void AddModule(Module *module) { m_modules.push_back(module); }

	const FunctionDefinition<VoidType> *GetEntryFunction(const std::string& name) const
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
	Function *GetEntryFunction(const std::string& name)
	{
		for (auto& module : m_modules)
		{
			if (module->ContainsEntryFunction(name))
			{
				return module->GetEntryFunction(name);
			}
		}
		Utils::Logger::LogError("Cannot find entry fuction '" + name);
	}

	// Formatting

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

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor& visitor) override
	{
		if (visitor.VisitIn(this))
		{
			for (auto& module : m_modules)
			{
				module->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

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

protected:
	std::vector<Module *> m_modules;
};

}
