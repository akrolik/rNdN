#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/HalfModifier.h"
#include "PTX/Instructions/Modifiers/RoundingModifier.h"
#include "PTX/Instructions/Modifiers/SaturateModifier.h"

namespace PTX {

template<class T, bool Assert = true>
class MultiplyInstruction : public InstructionBase_2<T>, public HalfModifier<T>, public RoundingModifier<T>, public FlushSubnormalModifier<T>, public SaturateModifier<T>
{
public:
	REQUIRE_TYPE_PARAM(MultiplyInstruction,
		REQUIRE_EXACT(T,
			Int16Type, Int32Type, Int64Type,
			UInt16Type, UInt32Type, UInt64Type,
			Float16Type, Float16x2Type, Float32Type, Float64Type
		)
	);

	template <class T1 = T, class = typename std::enable_if_t<HalfModifier<T1>::Enabled>>
	MultiplyInstruction(const Register<T> *destination, const TypedOperand<T> *sourceA, const TypedOperand<T> *sourceB, typename HalfModifier<T1>::Half half) : InstructionBase_2<T>(destination, sourceA, sourceB), HalfModifier<T>(half) {} 

	template <class T1 = T, class = typename std::enable_if_t<!HalfModifier<T1>::Enabled>>
	MultiplyInstruction(const Register<T> *destination, const TypedOperand<T> *sourceA, const TypedOperand<T> *sourceB) : InstructionBase_2<T>(destination, sourceA, sourceB) {}

	static std::string Mnemonic() { return "mul"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic();
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
			code += SaturateModifier<T>::OpCodeModifier();
		}
		return code + T::Name();
	}
};

}
