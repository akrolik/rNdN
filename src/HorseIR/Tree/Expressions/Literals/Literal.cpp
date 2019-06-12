#include "HorseIR/Tree/Expressions/Literals/Literal.h"

#include "HorseIR/Tree/Expressions/Literals/FunctionLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"

namespace HorseIR {

bool Literal::operator==(const Literal& other) const
{
	if (m_kind == other.m_kind)
	{
		switch (m_kind)
		{
			case Kind::Function:
				return static_cast<const FunctionLiteral&>(*this) == static_cast<const FunctionLiteral&>(other);
			case Kind::Vector:
				return static_cast<const VectorLiteral&>(*this) == static_cast<const VectorLiteral&>(other);
		}
	}
	return false;
}

bool Literal::operator!=(const Literal& other) const
{
	return !(*this == other);
}

}
