#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Assert = true>
class CNotInstruction : public InstructionBase_2<T>
{
public:
	REQUIRE_TYPE_PARAM(CNotInstruction,
		REQUIRE_EXACT(T,
			PredicateType, Bit16Type, Bit32Type, Bit64Type
		)
	);

	using InstructionBase_2<T>::InstructionBase_2;

	std::string OpCode() const override
	{
		return "cnot" + T::Name();
	}
};

}
