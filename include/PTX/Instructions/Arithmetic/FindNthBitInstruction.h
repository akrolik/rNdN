#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Typecheck = true>
class FindNthBitInstruction : public InstructionBase_3<Bit32Type, Bit32Type, T, Int32Type>
{
public:
	REQUIRE_TYPE(FindNthBitInstruction,
		Bit32Type, Int32Type, UInt32Type
	);

	using InstructionBase_3<Bit32Type, Bit32Type, T, Int32Type>::InstructionBase_3;

	std::string OpCode() const override
	{
		return "fns" + T::Name();
	}
};

}
