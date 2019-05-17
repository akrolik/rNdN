#pragma once

#include <string>

#include "HorseIR/Tree/Expressions/Operand.h"
#include "HorseIR/Tree/Expressions/LValue.h"

#include "HorseIR/Tree/Declaration.h"

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

	bool HasModule() const { return (m_module != ""); }

	const std::string& GetModule() const { return m_module; }
	void SetModule(const std::string& module) { m_module = module; }

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	Declaration *GetDeclaration() const { return m_declaration; }
	void SetDeclaration(Declaration *declaration) { m_declaration = declaration; }

	bool operator==(const Identifier& other) const
	{
		return (m_module == other.m_module && m_name == other.m_name && m_declaration == other.m_declaration);
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

	Declaration *m_declaration = nullptr;
};

}
