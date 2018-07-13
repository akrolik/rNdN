#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/CarryModifier.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/HalfModifier.h"
#include "PTX/Instructions/Modifiers/RoundingModifier.h"
#include "PTX/Instructions/Modifiers/SaturateModifier.h"

namespace PTX {

template<class T, bool Assert = true>
class MADInstruction : public InstructionBase_3<T>, public HalfModifier<T>, public RoundingModifier<T>, public FlushSubnormalModifier<T>, public SaturateModifier<T>, public CarryModifier<T>
{
public:
	REQUIRE_TYPE_PARAM(MADInstruction,
		REQUIRE_EXACT(T,
			Int16Type, Int32Type, Int64Type,
			UInt16Type, UInt32Type, UInt64Type,
			Float32Type, Float64Type
		)
	);

	using InstructionBase_3<T>::InstructionBase_3;

	static std::string Mnemonic() { return "mad"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic();
		if constexpr(CarryModifier<T>::Enabled)
		{
			code += CarryModifier<T>::OpCodeModifier();
		}
		if constexpr(HalfModifier<T>::Enabled)
		{
			code += HalfModifier<T>::OpCodeModifier();
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
			// Only applies in .hi mode
			if constexpr(HalfModifier<T>::Enabled)
			{
				if (HalfModifier<T>::GetUpper())
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
			}
			else
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
		}
		return code + T::Name();
	}
};

}
