#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/CarryModifier.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/HalfModifier.h"
#include "PTX/Instructions/Modifiers/RoundingModifier.h"
#include "PTX/Instructions/Modifiers/SaturateModifier.h"

namespace PTX {

template<class T, bool Typecheck = true>
class MADInstruction : public InstructionBase_3<T>, public HalfModifier<T>, public RoundingModifier<T>, public FlushSubnormalModifier<T>, public SaturateModifier<T>, public CarryModifier<T>
{
public:
	REQUIRE_TYPE(MADInstruction,
		Int16Type, Int32Type, Int64Type,
		UInt16Type, UInt32Type, UInt64Type,
		Float32Type, Float64Type
	);

	using InstructionBase_3<T>::InstructionBase_3;

	std::string OpCode() const override
	{
		std::string code = "mad";
		if constexpr(T::CarryModifier)
		{
			code += CarryModifier<T>::OpCodeModifier();
		}
		if constexpr(T::HalfModifier)
		{
			code += HalfModifier<T>::OpCodeModifier();
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
			// Only applies in .hi mode
			if constexpr(T::HalfModifier)
			{
				if (HalfModifier<T>::GetUpper())
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
			}
			else
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
		}
		return code + T::Name();
	}
};

}
