#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Typecheck = true>
class CountLeadingZerosInstruction : public InstructionBase_1<UInt32Type, T>
{
public:
	REQUIRE_TYPE(CountLeadingZerosInstruction,
		Bit32Type, Bit64Type
	);

	using InstructionBase_1<UInt32Type, T>::InstructionBase;

	std::string OpCode() const override
	{
		return "clz" + T::Name();
	}
};

}
