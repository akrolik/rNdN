#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

template<class T, bool Assert = true>
class AbsoluteInstruction : public InstructionBase_1<T>, public FlushSubnormalModifier<T>
{
public:
	REQUIRE_TYPE_PARAM(AbsoluteInstruction,
		REQUIRE_EXACT(T,
			Int16Type, Int32Type, Int64Type,
			Float32Type, Float64Type
		)
	);

	using InstructionBase_1<T>::InstructionBase_1;

	static std::string Mnemonic() { return "abs"; }
  	
	std::string OpCode() const override
	{
		if constexpr(FlushSubnormalModifier<T>::Enabled)
		{
			return Mnemonic() + FlushSubnormalModifier<T>::OpCodeModifier() + T::Name();
		}
		else
		{
			return Mnemonic() + T::Name();
		}
	}
};

}
