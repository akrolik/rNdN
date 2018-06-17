#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/CarryModifier.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/RoundingModifier.h"
#include "PTX/Instructions/Modifiers/SaturateModifier.h"

namespace PTX {

template<class T, bool Typecheck = true>
class AddInstruction : public InstructionBase_2<T>, public SaturateModifier<T>, public RoundingModifier<T>, public FlushSubnormalModifier<T>, public CarryModifier<T>
{
public:
	REQUIRE_TYPE(AddInstruction,
		Int16Type, Int32Type, Int64Type,
		UInt16Type, UInt32Type, UInt64Type,
		Float16Type, Float16x2Type, Float32Type, Float64Type
	);

	using InstructionBase_2<T>::InstructionBase_2;

	std::string OpCode() const override
	{
		std::string code = "add";
		if constexpr(T::CarryModifier)
		{
			code += CarryModifier<T>::OpCodeModifier();
		}
		if constexpr(is_rounding_type<T>::value)
		{
			code += RoundingModifier<T>::OpCodeModifier();
		}
		if constexpr(T::FlushModifier)
		{
			code += FlushSubnormalModifier<T>::OpCodeModifier();
		}
		if constexpr(T::SaturateModifier)
		{
			if constexpr(T::CarryModifier)
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
