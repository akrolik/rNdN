#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Assert = true>
class AndInstruction : public InstructionBase_2<T>
{
public:
	REQUIRE_TYPE_PARAM(AndInstruction,
		REQUIRE_EXACT(T,
			PredicateType, Bit16Type, Bit32Type, Bit64Type
		)
	);

	using InstructionBase_2<T>::InstructionBase_2;

	static std::string Mnemonic() { return "and"; }

	std::string OpCode() const override
	{
		return Mnemonic() + T::Name();
	}
};

}
