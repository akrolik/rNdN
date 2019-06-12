#include "HorseIR/Tree/Expressions/Operand.h"

#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/Literals/Literal.h"

namespace HorseIR {

bool Operand::operator==(const Operand& other) const
{
	if (m_kind == other.m_kind)
	{
		switch (m_kind)
		{
			case Kind::Identifier:
				return static_cast<const Identifier&>(*this) == static_cast<const Identifier&>(other);
			case Kind::Literal:
				return static_cast<const Literal&>(*this) == static_cast<const Literal&>(other);
		}
	}
	return false;
}

bool Operand::operator!=(const Operand& other) const
{
	return !(*this == other);
}

}
