#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/CarryModifier.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/RoundingModifier.h"
#include "PTX/Instructions/Modifiers/SaturateModifier.h"

namespace PTX {

template<class T, bool Typecheck = true>
class SubtractInstruction : public InstructionBase_2<T>, public SaturateModifier<T>, public RoundingModifier<T>, public FlushSubnormalModifier<T>, public CarryModifier<T>
{
public:
	REQUIRE_TYPE(SubtractInstruction,
		Int16Type, Int32Type, Int64Type,
		UInt16Type, UInt32Type, UInt64Type,
		Float16Type, Float16x2Type, Float32Type, Float64Type
	);

	using InstructionBase_2<T>::InstructionBase_2;

	std::string OpCode() const override
	{
		std::string code = "sub";
		if constexpr(CarryModifier<T>::Enabled)
		{
			code += CarryModifier<T>::OpCodeModifier();
		}
		if constexpr(RoundingModifier<T>::Enabled)
		{
			code += RoundingModifier<T>::OpCodeModifier();
		}
		if constexpr(FlushSubnormalModifier<T>::Enabled)
		{
			code += FlushSubnormalModifier<T>::OpCodeModifier();
		}
		if constexpr(SaturateModifier<T>::Enabled)
		{
			if constexpr(CarryModifier<T>::Enabled)
			{
				if (!CarryModifier<T>::IsActive())
				{
					code += SaturateModifier<T>::OpCodeModifier();
				}
			}
			else
			{
				code += SaturateModifier<T>::OpCodeModifier();
			}
		}
		return code + T::Name();
	}
};

}
