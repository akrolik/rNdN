#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

template<class T, bool Typecheck = true>
class MinimumInstruction : public InstructionBase_2<T>, public FlushSubnormalModifier<T>
{
public:
	REQUIRE_TYPE(MinimumInstruction,
		Int16Type, Int32Type, Int64Type,
		UInt16Type, UInt32Type, UInt64Type,
		Float32Type, Float64Type
	);

	using InstructionBase_2<T>::InstructionBase_2;

	std::string OpCode() const override
	{
		if constexpr(FlushSubnormalModifier<T>::Enabled)
		{
			return "min" + FlushSubnormalModifier<T>::OpCodeModifier() + T::Name();
		}
		else
		{
			return "min" + T::Name();
		}
	}
};

}
