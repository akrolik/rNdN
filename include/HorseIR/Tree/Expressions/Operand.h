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
	bool operator!=(const Operand& other) const
	{
		return !(*this == other);
	}

protected:
	Operand(Kind kind) : m_kind(kind) {}
	Kind m_kind;
};

}

#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/Literals/Literal.h"

namespace HorseIR {

inline bool Operand::operator==(const Operand& other) const
{
	if (m_kind == other.m_kind)
	{
		switch (m_kind)
		{
			case Kind::Identifier:
				return static_cast<const Identifier&>(*this) == static_cast<const Identifier&>(other);
			// case Kind::Literal:
				// return static_cast<const Literal&>(*this) == static_cast<const Literal&>(other);
		}
	}
	return false;
}

}
