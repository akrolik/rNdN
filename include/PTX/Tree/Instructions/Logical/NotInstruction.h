#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Assert = true>
class NotInstruction : public InstructionBase_1<T>
{
public:
	REQUIRE_TYPE_PARAM(NotInstruction,
		REQUIRE_EXACT(T,
			PredicateType, Bit16Type, Bit32Type, Bit64Type
		)
	);

	using InstructionBase_1<T>::InstructionBase_1;

	static std::string Mnemonic() { return "not"; }

	std::string OpCode() const override
	{
		return Mnemonic() + T::Name();
	}
};

}
