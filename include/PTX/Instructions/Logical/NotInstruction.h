#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Typecheck = true>
class NotInstruction : public InstructionBase_2<T>
{
public:
	REQUIRE_TYPE(NotInstruction,
		PredicateType, Bit16Type, Bit32Type, Bit64Type
	);

	using InstructionBase_2<T>::InstructionBase_2;

	std::string OpCode() const override
	{
		return "not" + T::Name();
	}
};

}
