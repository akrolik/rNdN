#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Typecheck = true>
class AndInstruction : public InstructionBase_2<T>
{
public:
	REQUIRE_TYPE(AndInstruction,
		PredicateType, Bit16Type, Bit32Type, Bit64Type
	);

	using InstructionBase_2<T>::InstructionBase_2;

	std::string OpCode() const override
	{
		return "and" + T::Name();
	}
};

}
