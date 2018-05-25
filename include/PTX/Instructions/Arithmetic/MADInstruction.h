#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/HalfModifier.h"
#include "PTX/Instructions/Modifiers/RoundingModifier.h"
#include "PTX/Instructions/Modifiers/SaturateModifier.h"

namespace PTX {

template<class T>
class MADInstruction : public InstructionBase<T, 3>, public HalfModifier<T>, public RoundingModifier<T>, public FlushSubnormalModifier<T>, public SaturateModifier<T>
{
	REQUIRE_BASE_TYPE(MADInstruction, ScalarType);
	DISABLE_EXACT_TYPE(MADInstruction, Int8Type);
	DISABLE_EXACT_TYPE(MADInstruction, UInt8Type);
	DISABLE_EXACT_TYPE(MADInstruction, Float16Type);
	DISABLE_EXACT_TYPE(MADInstruction, Float16x2Type);
	DISABLE_EXACT_TYPE_TEMPLATE(MADInstruction, BitType);
public:
	using InstructionBase<T, 3>::InstructionBase;

	std::string OpCode() const
	{
		std::string code = "mad";
		if constexpr(T::HalfModifier)
		{
			if (this->m_lower)
			{
				code += ".lo";
			}
			else if (this->m_upper)
			{
				code += ".hi";

				// Only applies in .hi mode
				if constexpr(T::SaturateModifier)
				{
					if (this->m_saturate)
					{
						code += ".sat";
					}
				}
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
		if constexpr(T::SaturateModifier && !T::HalfModifier)
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
