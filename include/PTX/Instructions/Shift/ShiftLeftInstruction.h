#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Typecheck = true>
class ShiftLeftInstruction : public InstructionBase_2<T, T, UInt32Type>
{
public:
	REQUIRE_TYPE(ShiftLeftInstruction,
		Bit16Type, Bit32Type, Bit64Type
	);

	using InstructionBase_2<T, T, UInt32Type>::InstructionBase_2;

	std::string OpCode() const override
	{
		return "shl" + T::Name();
	}
};

}
