#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Assert = true>
class ShiftLeftInstruction : public InstructionBase_2<T, T, UInt32Type>
{
public:
	REQUIRE_TYPE_PARAM(ShiftLeftInstruction,
		REQUIRE_EXACT(T,
			Bit16Type, Bit32Type, Bit64Type
		)
	);

	using InstructionBase_2<T, T, UInt32Type>::InstructionBase_2;

	static std::string Mnemonic() { return "shl"; }

	std::string OpCode() const override
	{
		return Mnemonic() + T::Name();
	}
};

}
