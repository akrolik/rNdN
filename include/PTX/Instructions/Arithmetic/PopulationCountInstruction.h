#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Assert = true>
class PopulationCountInstruction : public InstructionBase_1<UInt32Type, T>
{
public:
	REQUIRE_TYPE_PARAM(PopulationCountInstruction,
		REQUIRE_EXACT(T,
			Bit32Type, Bit64Type
		)
	);

	using InstructionBase_1<UInt32Type, T>::InstructionBase_1;

	static std::string Mnemonic() { return "popc"; }

	std::string OpCode() const override
	{
		return Mnemonic() + T::Name();
	}
};

}
