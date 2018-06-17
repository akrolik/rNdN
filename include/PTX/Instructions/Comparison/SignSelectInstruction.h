#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

template<class D, class T, bool Typecheck = true>
class SignSelectInstruction : public InstructionBase_3<D, D, D, T>, public FlushSubnormalModifier<T>
{
public:
	REQUIRE_TYPE_PARAMS(SetInstruction,
		REQUIRE_TYPE_PARAM(D,
			Bit16Type, Bit32Type, Bit64Type,
			Int16Type, Int32Type, Int64Type,
			UInt16Type, UInt32Type, UInt64Type,
			Float32Type, Float64Type
		),
		REQUIRE_TYPE_PARAM(T,
			Int32Type, Float32Type
		)
	);

	using InstructionBase_3<D, D, D, T>::InstructionBase_3;

	std::string OpCode() const override
	{
		std::string code = "slct";
		if constexpr(T::FlushModifier)
		{
			code += FlushSubnormalModifier<T>::OpCodeModifier();
		}
		return code + D::Name() + T::Name();
	}
};

}
