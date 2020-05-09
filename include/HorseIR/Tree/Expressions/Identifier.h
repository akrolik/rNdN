#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/LValue.h"
#include "HorseIR/Tree/Expressions/Operand.h"

#include "HorseIR/Tree/VariableDeclaration.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class Identifier : public Operand, public LValue
{
public:
	Identifier(const std::string& name) : Operand(Operand::Kind::Identifier), m_module(""), m_name(name) {}
	Identifier(const std::string& module, const std::string& name) : Operand(Operand::Kind::Identifier), m_module(module), m_name(name) {}

	Identifier *Clone() const override
	{
		return new Identifier(m_module, m_name);
	}

	bool HasModule() const { return (m_module != ""); }

	std::string GetFullName() const { return m_module + "." + m_name; }

	const std::string& GetModule() const { return m_module; }
	void SetModule(const std::string& module) { m_module = module; }

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	// LValue
	Type *GetType() const override { return Operand::GetType(); }

	bool operator==(const Identifier& other) const
	{
		return (m_module == other.m_module && m_name == other.m_name);
	}

	bool operator!=(const Identifier& other) const
	{
		return !(*this == other);
	}
	
	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

protected:
	std::string m_module = "";
	std::string m_name;
};

}
