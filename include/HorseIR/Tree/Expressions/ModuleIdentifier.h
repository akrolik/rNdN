#pragma once

#include <string>

#include "HorseIR/Tree/Expressions/Expression.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class ModuleIdentifier : public Expression
{
public:
	ModuleIdentifier(const std::string& name) : m_module(""), m_name(name) {}
	ModuleIdentifier(const std::string& module, const std::string& name) : m_module(module), m_name(name) {}

	const Type *GetType() const { return m_type; }
	void SetType(Type *type) { m_type = type; }

	const std::string& GetModule() const { return m_module; }
	const std::string& GetName() const { return m_name; }

	std::string ToString() const override
	{
		if (m_module != "")
		{
			return m_module + "." + m_name;
		}
		return m_name;
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

private:
	std::string m_module = "";
	std::string m_name;
	Type *m_type = nullptr;
};

}
