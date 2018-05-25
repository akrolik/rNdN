#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/RoundingModifier.h"
#include "PTX/Instructions/Modifiers/SaturateModifier.h"

namespace PTX {

template<class T>
class AddInstruction : public InstructionBase<T, 2>, public SaturateModifier<T>, public RoundingModifier<T>, public FlushSubnormalModifier<T>
{
	REQUIRE_BASE_TYPE(AddInstruction, ScalarType);
	DISABLE_EXACT_TYPE(AddInstruction, Int8Type);
	DISABLE_EXACT_TYPE(AddInstruction, UInt8Type);
	DISABLE_EXACT_TYPE_TEMPLATE(AddInstruction, BitType);
public:
	using InstructionBase<T, 2>::InstructionBase;

	std::string OpCode() const
	{
		std::string code = "add";
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
