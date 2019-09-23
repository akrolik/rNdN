#pragma once

#include <vector>

#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/SymbolValue.h"

#include "HorseIR/Tree/Types/BasicType.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class SymbolLiteral : public TypedVectorLiteral<SymbolValue *>
{
public:
	SymbolLiteral(SymbolValue *value) : TypedVectorLiteral<SymbolValue *>(value, BasicType::BasicKind::Symbol) {}
	SymbolLiteral(const std::vector<SymbolValue *>& values) : TypedVectorLiteral<SymbolValue *>(values, BasicType::BasicKind::Symbol) {}

	SymbolLiteral *Clone() const override
	{
		std::vector<SymbolValue *> values;
		for (const auto& value : m_values)
		{
			values.push_back(value->Clone());
		}
		return new SymbolLiteral(values);
	}

	bool operator==(const SymbolLiteral& other) const
	{
		return std::equal(
			std::begin(m_values), std::end(m_values),
			std::begin(other.m_values), std::end(other.m_values),
			[](const SymbolValue *v1, const SymbolValue *v2) { return *v1 == *v2; }
		);
	}

	bool operator!=(const SymbolLiteral& other) const
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
