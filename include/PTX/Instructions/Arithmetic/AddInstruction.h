#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/CarryModifier.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/RoundingModifier.h"
#include "PTX/Instructions/Modifiers/SaturateModifier.h"

namespace PTX {

template<class T>
class AddInstruction : public InstructionBase_2<T>, public SaturateModifier<T>, public RoundingModifier<T>, public FlushSubnormalModifier<T>, public CarryModifier<T>
{
	REQUIRE_BASE_TYPE(AddInstruction, ScalarType);
	DISABLE_EXACT_TYPE(AddInstruction, Int8Type);
	DISABLE_EXACT_TYPE(AddInstruction, UInt8Type);
	DISABLE_EXACT_TYPE_TEMPLATE(AddInstruction, BitType);
public:
	using InstructionBase_2<T>::InstructionBase_2;

	std::string OpCode() const override
	{
		std::string code = "add";
		if constexpr(T::CarryModifier)
		{
			if (this->m_carryIn)
			{
				code += "c";
			}
			if (this->m_carryOut)
			{
				code += ".cc";
			}
		}
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
			if constexpr(T::CarryModifier)
			{
				if (!this->m_carryIn && !this->m_carryOut && this->m_saturate)
				{
					code += ".sat";
				}
			}
			else
			{
				if (this->m_saturate)
				{
					code += ".sat";
				}
			}
		}
		return code + T::Name();
	}
};

}
