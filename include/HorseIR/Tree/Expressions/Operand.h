#pragma once

#include "HorseIR/Tree/Expressions/Expression.h"

namespace HorseIR {

class Operand : public Expression
{
public:
	bool operator==(const Operand& other) const;
	bool operator!=(const Operand& other) const;

protected:
	enum class Kind {
		Identifier,
		Literal
	};

	Operand(Kind kind) : m_kind(kind) {}
	Kind m_kind;
};

}
