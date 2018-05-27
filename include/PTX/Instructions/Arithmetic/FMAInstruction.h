#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class FMAInstruction : public InstructionBase_3<T>, public RoundingModifier<T, true>, public FlushSubnormalModifier<T>, public SaturateModifier<T>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(FMAInstruction, FloatType);
public:
	FMAInstruction(Register<T> *destination, Operand<T> *sourceA, Operand<T> *sourceB, Operand<T> *sourceC, typename T::RoundingMode roundingMode) : InstructionBase_3<T>(destination, sourceA, sourceB, sourceC), RoundingModifier<T, true>(roundingMode) {}

	std::string OpCode() const
	{
		std::string code = "fma" + T::RoundingModeString(this->m_roundingMode);
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
