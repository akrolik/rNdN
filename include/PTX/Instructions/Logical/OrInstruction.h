#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Typecheck = true>
class OrInstruction : public InstructionBase_2<T>
{
public:
	REQUIRE_TYPE(OrInstruction,
		PredicateType, Bit16Type, Bit32Type, Bit64Type
	);

	using InstructionBase_2<T>::InstructionBase_2;

	std::string OpCode() const override
	{
		return "or" + T::Name();
	}
};

}
