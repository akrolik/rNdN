#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

template<class T, class S, bool Typecheck = true>
class SignSelectInstruction : public InstructionBase_3<T, T, T, S>, public FlushSubnormalModifier<T>
{
	//TODO: macro this
	static_assert(
		std::is_same<S, Int32Type>::value ||
		std::is_same<S, Float32Type>::value,
		"PTX::SignSelectInstruction requires a signed 32-bit value"
	);
public:
	REQUIRE_TYPE(SignSelectInstruction,
		Bit16Type, Bit32Type, Bit64Type,
		Int16Type, Int32Type, Int64Type,
		UInt16Type, UInt32Type, UInt64Type,
		Float32Type, Float64Type
	);

	using InstructionBase_3<T, T, T, S>::InstructionBase_3;

	std::string OpCode() const override
	{
		std::string code = "slct";
		if constexpr(T::FlushModifier)
		{
			if (this->m_flush)
			{
				code += ".ftz";
			}
		}
		return code + T::Name() + S::Name();
	}
};

}
