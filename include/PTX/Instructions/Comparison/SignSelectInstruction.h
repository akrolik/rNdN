#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

template<class T, class S>
class SignSelectInstruction : public InstructionBase_3<T, T, T, S>, public FlushSubnormalModifier<T>
{
	static_assert(
		std::is_same<S, Int32Type>::value ||
		std::is_same<S, Float32Type>::value,
		"PTX::SignSelectInstruction requires a signed 32-bit value"
	);
	REQUIRE_BASE_TYPE(SignSelectInstruction, ScalarType);
	DISABLE_EXACT_TYPE(SignSelectInstruction, PredicateType);
	DISABLE_EXACT_TYPE(SignSelectInstruction, Bit8Type);
	DISABLE_EXACT_TYPE(SignSelectInstruction, Int8Type);
	DISABLE_EXACT_TYPE(SignSelectInstruction, UInt8Type);
	DISABLE_EXACT_TYPE(SignSelectInstruction, Float16Type);
	DISABLE_EXACT_TYPE(SignSelectInstruction, Float16x2Type);
public:
	using InstructionBase_3<T, T, T, S>::InstructionBase_3;

	std::string OpCode() const
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
