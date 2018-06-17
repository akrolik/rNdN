#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Typecheck = true>
class ShiftRightInstruction : public InstructionBase_2<T, T, UInt32Type>
{
public:
	REQUIRE_TYPE(ShiftRightInstruction,
		Bit16Type, Bit32Type, Bit64Type
	);

	using InstructionBase_2<T, T, UInt32Type>::InstructionBase_2;

	std::string OpCode() const override
	{
		return "shr" + T::Name();
	}
};

}
