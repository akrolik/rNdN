#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"

#include "HorseIR/Tree/Types/BasicType.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class StringLiteral : public TypedVectorLiteral<std::string>
{
public:
	StringLiteral(const std::string& value) : TypedVectorLiteral<std::string>(value, BasicType::BasicKind::String) {}
	StringLiteral(const std::vector<std::string>& values) : TypedVectorLiteral<std::string>(values, BasicType::BasicKind::String) {}

	bool operator==(const StringLiteral& other) const
	{
		return (m_values == other.m_values);
	}

	bool operator!=(const StringLiteral& other) const
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
};

}
