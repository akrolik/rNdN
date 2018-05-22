#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class PopulationCountInstruction : public InstructionBase<T, 1, UInt32Type>
{
	REQUIRE_TYPES(PopulationCountInstruction, BitType);
	DISABLE_TYPE(PopulationCountInstruction, Bit8Type);
	DISABLE_TYPE(PopulationCountInstruction, Bit16Type);
public:
	using InstructionBase<T, 1, UInt32Type>::InstructionBase;

	std::string OpCode() const
	{
		return "popc" + T::Name();
	}
};

}
