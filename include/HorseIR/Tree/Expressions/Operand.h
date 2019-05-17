#pragma once

#include "HorseIR/Tree/Expressions/Expression.h"

namespace HorseIR {

class Operand : public Expression
{
public:
	enum class Kind {
		Identifier,
		Literal
	};

	bool operator==(const Operand& other) const;
	bool operator!=(const Operand& other) const;

protected:
	Operand(Kind kind) : m_kind(kind) {}
	Kind m_kind;
};

}
