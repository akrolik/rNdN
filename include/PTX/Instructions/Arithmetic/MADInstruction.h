#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/CarryModifier.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/HalfModifier.h"
#include "PTX/Instructions/Modifiers/RoundingModifier.h"
#include "PTX/Instructions/Modifiers/SaturateModifier.h"

namespace PTX {

template<class T>
class MADInstruction : public InstructionBase_3<T>, public HalfModifier<T>, public RoundingModifier<T>, public FlushSubnormalModifier<T>, public SaturateModifier<T>, public CarryModifier<T>
{
	REQUIRE_BASE_TYPE(MADInstruction, ScalarType);
	DISABLE_EXACT_TYPE(MADInstruction, Int8Type);
	DISABLE_EXACT_TYPE(MADInstruction, UInt8Type);
	DISABLE_EXACT_TYPE(MADInstruction, Float16Type);
	DISABLE_EXACT_TYPE(MADInstruction, Float16x2Type);
	DISABLE_EXACT_TYPE_TEMPLATE(MADInstruction, BitType);
public:
	using InstructionBase_3<T>::InstructionBase_3;

	std::string OpCode() const override
	{
		std::string code = "mad";
		if constexpr(T::CarryModifier)
		{
			if (this->m_carryIn)
			{
				code += "c";
			}
		}
		if constexpr(T::HalfModifier)
		{
			if (this->m_lower)
			{
				code += ".lo";
			}
			else if (this->m_upper)
			{
				code += ".hi";
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
			// Only applies in .hi mode
			if constexpr(T::HalfModifier)
			{
				if constexpr(T::CarryModifier)
				{
					if (!this->m_carryIn && !this->m_carryOut && this->m_upper && this->m_saturate)
					{
						code += ".sat";

					}
				}
				else
				{
					if (this->m_upper && this->m_saturate)
					{
						code += ".sat";
					}
				}
			}
			else
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
		}
		return code + T::Name();
	}
};

}
