#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Typecheck = true>
class BitReverseInstruction : public InstructionBase_2<T>
{
public:
	REQUIRE_TYPE(BitReverseInstruction,
		Bit32Type, Bit64Type
	);

	using InstructionBase_2<T>::InstructionBase;

	std::string OpCode() const override
	{
		return "brev" + T::Name();
	}
};

}
