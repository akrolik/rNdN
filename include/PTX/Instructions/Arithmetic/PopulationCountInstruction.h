#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class PopulationCountInstruction : public InstructionBase_1<UInt32Type, T>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(PopulationCountInstruction, BitType);
	DISABLE_EXACT_TYPE(PopulationCountInstruction, PredicateType);
	DISABLE_EXACT_TYPE(PopulationCountInstruction, Bit8Type);
	DISABLE_EXACT_TYPE(PopulationCountInstruction, Bit16Type);
public:
	using InstructionBase_1<UInt32Type, T>::InstructionBase_1;

	std::string OpCode() const
	{
		return "popc" + T::Name();
	}
};

}
