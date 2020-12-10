#pragma once

#include "PTX/Tree/Node.h"

#include <string>
#include <vector>
#include <unordered_map>

#include "PTX/Tree/Declarations/Declaration.h"
#include "PTX/Tree/Directives/Directive.h"
#include "PTX/Tree/Functions/Function.h"
#include "PTX/Tree/Type.h"

namespace PTX {

class Module : public Node
{
public:
	// Properties

	unsigned int GetMajorVersion() const { return m_versionMajor; }
	unsigned int GetMinorVersion() const { return m_versionMinor; }
	void SetVersion(unsigned int major, unsigned int minor) { m_versionMajor = major; m_versionMinor = minor; }

	const std::string& GetTarget() const { return m_target; }
	void SetDeviceTarget(const std::string& target) { m_target = target; }

	Bits GetAddressSize() const { return m_addressSize; }
	void SetAddressSize(Bits addressSize) { m_addressSize = addressSize; }

	// Entry function

	void AddEntryFunction(Function *function)
	{
		m_entryFunctions.insert({function->GetName(), function});
	}
	bool ContainsEntryFunction(const std::string& name) const
	{
		return m_entryFunctions.find(name) != m_entryFunctions.end();
	}
	const Function *GetEntryFunction(const std::string& name) const
	{
		return m_entryFunctions.at(name);
	}
	Function *GetEntryFunction(const std::string& name)
	{
		return m_entryFunctions.at(name);
	}

	const std::unordered_map<std::string, const Function *> GetEntryFunctions() const
	{
		return std::unordered_map<std::string, const Function *>(std::begin(m_entryFunctions), std::end(m_entryFunctions));
	}
	std::unordered_map<std::string, Function *>& GetEntryFunctions() { return m_entryFunctions; }

	// Contents

	void AddDirective(Directive *directive)
	{
		m_directives.push_back(directive);
	}
	template<class T>
	void AddDirectives(const std::vector<T>& directives)
	{
		m_directives.insert(std::end(m_directives), std::begin(directives), std::end(directives));
	}

	std::vector<const Directive *> GetDirectives() const
	{
		return std::vector<const Directive *>(std::begin(m_directives), std::end(m_directives));
	}
	std::vector<Directive *>& GetDirectives() { return m_directives; }

	void AddDeclaration(Declaration *declaration)
	{
		m_declarations.push_back(declaration);
	}
	template<class T>
	void AddDeclarations(const std::vector<T>& declarations)
	{
		m_declarations.insert(std::end(m_declarations), std::begin(declarations), std::end(declarations));
	}

	void InsertDeclaration(Declaration *declaration, unsigned int index)
	{
		m_declarations.insert(std::begin(m_declarations) + index, declaration);
	}
	template<class T>
	void InsertDeclarations(const std::vector<T>& declarations, unsigned int index)
	{
		m_declarations.insert(std::begin(m_declarations) + index, std::begin(declarations), std::end(declarations));
	}

	std::vector<const Declaration *> GetDeclarations() const
	{
		return std::vector<const Declaration *>(std::begin(m_declarations), std::end(m_declarations));
	}
	std::vector<Declaration *>& GetDeclarations() { return m_declarations; }

	// Formatting

	json ToJSON() const override
	{
		json j;
		j["version_major"] = m_versionMajor;
		j["version_minor"] = m_versionMinor;
		j["target"] = m_target;
		j["address_size"] = DynamicBitSize::GetBits(m_addressSize);
		for (const auto& directive : m_directives)
		{
			j["directive"].push_back(directive->ToJSON());
		}
		for (const auto& declaration : m_declarations)
		{
			j["declarations"].push_back(declaration->ToJSON());
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
			for (auto& directive : m_directives)
			{
				directive->Accept(visitor);
			}
			for (auto& declaration : m_declarations)
			{
				declaration->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor& visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			for (const auto& directive : m_directives)
			{
				directive->Accept(visitor);
			}
			for (const auto& declaration : m_declarations)
			{
				declaration->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

private:
	unsigned int m_versionMajor, m_versionMinor;
	std::string m_target;
	Bits m_addressSize = Bits::Bits32;

	std::vector<Directive *> m_directives;
	std::vector<Declaration *> m_declarations;

	std::unordered_map<std::string, Function *> m_entryFunctions;
};

}
