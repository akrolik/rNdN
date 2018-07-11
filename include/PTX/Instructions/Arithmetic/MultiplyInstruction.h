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

	using InstructionBase_2<T>::InstructionBase_2;

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

template<>
class MultiplyInstruction<Int32Type> : public InstructionBase_2<Int32Type>, public HalfModifier<Int32Type>
{
public:
	using InstructionBase_2<Int32Type>::InstructionBase_2;

	std::string OpCode() const override
	{
		return "mul" + HalfModifier<Int32Type>::OpCodeModifier() + Int32Type::Name();
	}
};

}
