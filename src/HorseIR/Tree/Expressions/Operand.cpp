#include "HorseIR/Tree/Expressions/Operand.h"

#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/Literals/Literal.h"

#include "HorseIR/TypeUtils.h"
#include "HorseIR/Tree/Types/Type.h"
#include "HorseIR/Tree/Types/BasicType.h"

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
			{
				if (*m_type != *other.m_type)
				{
					return false;
				}
				auto type = HorseIR::template GetType<BasicType>(m_type);
				switch (type->GetKind())
				{
					case BasicType::Kind::Bool:
						return static_cast<const Literal<bool>&>(*this) == static_cast<const Literal<bool>&>(other);
					case BasicType::Kind::Int8:
						return static_cast<const Literal<std::int8_t>&>(*this) == static_cast<const Literal<std::int8_t>&>(other);
					case BasicType::Kind::Int16:
						return static_cast<const Literal<std::int16_t>&>(*this) == static_cast<const Literal<std::int16_t>&>(other);
					case BasicType::Kind::Int32:
						return static_cast<const Literal<std::int32_t>&>(*this) == static_cast<const Literal<std::int32_t>&>(other);
					case BasicType::Kind::Int64:
						return static_cast<const Literal<std::int64_t>&>(*this) == static_cast<const Literal<std::int64_t>&>(other);
					case BasicType::Kind::Float32:
						return static_cast<const Literal<float>&>(*this) == static_cast<const Literal<float>&>(other);
					case BasicType::Kind::Float64:
						return static_cast<const Literal<double>&>(*this) == static_cast<const Literal<double>&>(other);
					case BasicType::Kind::Symbol:
					case BasicType::Kind::String:
						return static_cast<const Literal<std::string>&>(*this) == static_cast<const Literal<std::string>&>(other);
					case BasicType::Kind::Date:
						return static_cast<const Literal<std::int64_t>&>(*this) == static_cast<const Literal<std::int64_t>&>(other);
				}
			}
		}
	}
	return false;
}

bool Operand::operator!=(const Operand& other) const
{
	return !(*this == other);
}

}
