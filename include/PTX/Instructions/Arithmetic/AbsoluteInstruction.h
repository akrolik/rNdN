#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

template<class T, bool Typecheck = true>
class AbsoluteInstruction : public InstructionBase_1<T>, public FlushSubnormalModifier<T>
{
public:
	REQUIRE_TYPE(AbsoluteInstruction,
		Int16Type, Int32Type, Int64Type,
		Float32Type, Float64Type
	);

	using InstructionBase_1<T>::InstructionBase_1;
  	
	std::string OpCode() const override
	{
		if constexpr(T::FlushModifier)
		{
			return "abs" + FlushSubnormalModifier<T>::OpCodeModifier() + T::Name();
		}
		else
		{
			return "abs" + T::Name();
		}
	}
};

}
