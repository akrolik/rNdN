#pragma once

#include <string>

#include "HorseIR/Tree/Node.h"
#include "HorseIR/Tree/LValue.h"

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class VariableDeclaration : virtual public Node, public LValue
{
public:
	VariableDeclaration(const std::string& name, Type *type) : m_name(name), m_type(type) {}

	VariableDeclaration *Clone() const override
	{
		return new VariableDeclaration(m_name, m_type->Clone());
	}

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	// LValue
	Type *GetType() const override { return m_type; }
	void SetType(Type *type) { m_type = type; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			m_type->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			m_type->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

protected:
	std::string m_name;
	Type *m_type = nullptr;
};

}
