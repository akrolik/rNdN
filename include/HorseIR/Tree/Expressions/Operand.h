#pragma once

#include "HorseIR/Tree/Expressions/Expression.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class Operand : public Expression
{
public:
	enum class Kind {
		Identifier,
		Literal
	};

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	bool operator==(const Operand& other) const;
	bool operator!=(const Operand& other) const;

protected:
	Operand(Kind kind) : m_kind(kind) {}
	Kind m_kind;
};

}
