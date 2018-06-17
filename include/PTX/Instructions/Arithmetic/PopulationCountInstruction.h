#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Typecheck = true>
class PopulationCountInstruction : public InstructionBase_1<UInt32Type, T>
{
public:
	REQUIRE_TYPE(PopulationCountInstruction,
		Bit32Type, Bit64Type
	);

	using InstructionBase_1<UInt32Type, T>::InstructionBase_1;

	std::string OpCode() const override
	{
		return "popc" + T::Name();
	}
};

}
