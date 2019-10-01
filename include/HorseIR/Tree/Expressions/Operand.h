#pragma once

#include "HorseIR/Tree/Expressions/Expression.h"

namespace HorseIR {

class Operand : public Expression
{
public:
	virtual Operand *Clone() const override = 0;

	bool operator==(const Operand& other) const;
	bool operator!=(const Operand& other) const;

	// An operand only has a single type
	Type *GetType() const { return m_types.at(0); }

protected:
	enum class Kind {
		Identifier,
		Literal
	};

	Operand(Kind kind) : m_kind(kind) {}
	Kind m_kind;
};

}
