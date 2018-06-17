#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Typecheck = true>
class FMAInstruction : public InstructionBase_3<T>, public RoundingModifier<T, true>, public FlushSubnormalModifier<T>, public SaturateModifier<T>
{
public:
	REQUIRE_TYPE(FMAInstruction,
		Float16Type, Float16x2Type, Float32Type, Float64Type
	);

	FMAInstruction(const Register<T> *destination, const Operand<T> *sourceA, const Operand<T> *sourceB, const Operand<T> *sourceC, typename T::RoundingMode roundingMode) : InstructionBase_3<T>(destination, sourceA, sourceB, sourceC), RoundingModifier<T, true>(roundingMode) {}

	std::string OpCode() const override
	{
		std::string code = "fma" + RoundingModifier<T, true>::OpCodeModifier();
		if constexpr(T::FlushModifier)
		{
			code += FlushSubnormalModifier<T>::OpCodeModifier();
		}
		if constexpr(T::SaturateModifier)
		{
			code += SaturateModifier<T>::OpCodeModifier();
		}
		return code + T::Name();
	}
};

}
