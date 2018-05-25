#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/RoundingModifier.h"
#include "PTX/Instructions/Modifiers/SaturateModifier.h"

namespace PTX {

template<class T>
class SubtractInstruction : public InstructionBase<T, 2>, public SaturateModifier<T>, public RoundingModifier<T>, public FlushSubnormalModifier<T>
{
	REQUIRE_BASE_TYPE(SubtractInstruction, ScalarType);
	DISABLE_EXACT_TYPE(SubtractInstruction, Int8Type);
	DISABLE_EXACT_TYPE(SubtractInstruction, UInt8Type);
	DISABLE_EXACT_TYPE_TEMPLATE(SubtractInstruction, BitType);
public:
	using InstructionBase<T, 2>::InstructionBase;

	std::string OpCode() const
	{
		std::string code = "sub";
		if constexpr(is_rounding_type<T>::value)
		{
			code += T::RoundingModeString(this->m_roundingMode);
		}
		if constexpr(T::FlushModifier)
		{
			if (this->m_flush)
			{
				code += ".ftz";
			}
		}
		if constexpr(T::SaturateModifier)
		{
			if (this->m_saturate)
			{
				code += ".sat";
			}
		}
		return code + T::Name();
	}
};

}
