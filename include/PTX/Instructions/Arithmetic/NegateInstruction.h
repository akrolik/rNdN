#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

template<class T, bool Assert = true>
class NegateInstruction : public InstructionBase_1<T>, public FlushSubnormalModifier<T>
{
public:
	REQUIRE_TYPE_PARAM(NegateInstruction,
		REQUIRE_EXACT(T,
			Int16Type, Int32Type, Int64Type,
			Float16Type, Float16x2Type, Float32Type, Float64Type
		)
	);

	using InstructionBase_1<T>::InstructionBase_1;

	static std::string Mnemonic() { return "neg"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic();
		if constexpr(FlushSubnormalModifier<T>::Enabled)
		{
			code += FlushSubnormalModifier<T>::OpCodeModifier();
		}
		return code + T::Name();
	}
};

}
