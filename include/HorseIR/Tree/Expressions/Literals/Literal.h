#pragma once

#include "HorseIR/Tree/Expressions/Operand.h"

namespace HorseIR {

class Literal : public Operand
{
public:
	virtual Literal *Clone() const override = 0;

	// Operators

	bool operator==(const Literal& other) const;
	bool operator!=(const Literal& other) const;

protected:
	enum class Kind {
		Function,
		Vector
	};

	Literal(Kind kind) : Operand(Operand::Kind::Literal), m_kind(kind) {}
	Kind m_kind;
};

}
