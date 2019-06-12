#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"

#include "HorseIR/Tree/Expressions/Literals/BooleanLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/CharLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/Int8Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Int16Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Int32Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Int64Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Float32Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Float64Literal.h"
#include "HorseIR/Tree/Expressions/Literals/ComplexLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/SymbolLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/StringLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/DatetimeLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/DateLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/MonthLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/MinuteLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/SecondLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/TimeLiteral.h"

namespace HorseIR {

bool VectorLiteral::operator==(const VectorLiteral& other) const
{
	if (m_basicKind == other.m_basicKind)
	{
		switch (m_basicKind)
		{
			case BasicType::BasicKind::Boolean:
				return static_cast<const BooleanLiteral&>(*this) == static_cast<const BooleanLiteral&>(other);
			case BasicType::BasicKind::Char:
				return static_cast<const CharLiteral&>(*this) == static_cast<const CharLiteral&>(other);
			case BasicType::BasicKind::Int8:
				return static_cast<const Int8Literal&>(*this) == static_cast<const Int8Literal&>(other);
			case BasicType::BasicKind::Int16:
				return static_cast<const Int16Literal&>(*this) == static_cast<const Int16Literal&>(other);
			case BasicType::BasicKind::Int32:
				return static_cast<const Int32Literal&>(*this) == static_cast<const Int32Literal&>(other);
			case BasicType::BasicKind::Int64:
				return static_cast<const Int64Literal&>(*this) == static_cast<const Int64Literal&>(other);
			case BasicType::BasicKind::Float32:
				return static_cast<const Float32Literal&>(*this) == static_cast<const Float32Literal&>(other);
			case BasicType::BasicKind::Float64:
				return static_cast<const Float64Literal&>(*this) == static_cast<const Float64Literal&>(other);
			case BasicType::BasicKind::Complex:
				return static_cast<const ComplexLiteral&>(*this) == static_cast<const ComplexLiteral&>(other);
			case BasicType::BasicKind::Symbol:
				return static_cast<const SymbolLiteral&>(*this) == static_cast<const SymbolLiteral&>(other);
			case BasicType::BasicKind::String:
				return static_cast<const StringLiteral&>(*this) == static_cast<const StringLiteral&>(other);
			case BasicType::BasicKind::Datetime:
				return static_cast<const DatetimeLiteral&>(*this) == static_cast<const DatetimeLiteral&>(other);
			case BasicType::BasicKind::Date:
				return static_cast<const DateLiteral&>(*this) == static_cast<const DateLiteral&>(other);
			case BasicType::BasicKind::Month:
				return static_cast<const MonthLiteral&>(*this) == static_cast<const MonthLiteral&>(other);
			case BasicType::BasicKind::Minute:
				return static_cast<const MinuteLiteral&>(*this) == static_cast<const MinuteLiteral&>(other);
			case BasicType::BasicKind::Second:
				return static_cast<const SecondLiteral&>(*this) == static_cast<const SecondLiteral&>(other);
			case BasicType::BasicKind::Time:
				return static_cast<const TimeLiteral&>(*this) == static_cast<const TimeLiteral&>(other);
		}
	}
	return false;
}

bool VectorLiteral::operator!=(const VectorLiteral& other) const
{
	return !(*this == other);
}

}
