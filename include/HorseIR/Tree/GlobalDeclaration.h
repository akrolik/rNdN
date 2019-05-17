#pragma once

#include <string>

#include "HorseIR/Tree/ModuleContent.h"

#include "HorseIR/Tree/Types/Type.h"
#include "HorseIR/Tree/Expressions/Operand.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class GlobalDeclaration : public ModuleContent
{
public:
	GlobalDeclaration(const std::string& name, Type *type, Operand *expression) : m_name(name), m_type(type), m_expression(expression) {}

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	Type *GetType() const { return m_type; }
	void SetType(Type *type) { m_type = type; }

	Operand *GetExpression() const { return m_expression; }
	void SetExpression(Operand *expression) { m_expression = expression; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			m_type->Accept(visitor);
			m_expression->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			m_type->Accept(visitor);
			m_expression->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

protected:
	std::string m_name;
	Type *m_type = nullptr;
	Operand *m_expression = nullptr;
};

}
