#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class PopulationCountInstruction : public InstructionBase<T, 1, UInt32Type>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(PopulationCountInstruction, BitType);
	DISABLE_EXACT_TYPE(PopulationCountInstruction, PredicateType);
	DISABLE_EXACT_TYPE(PopulationCountInstruction, Bit8Type);
	DISABLE_EXACT_TYPE(PopulationCountInstruction, Bit16Type);
public:
	using InstructionBase<T, 1, UInt32Type>::InstructionBase;

	std::string OpCode() const
	{
		return "popc" + T::Name();
	}
};

}
