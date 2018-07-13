#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Assert = true>
class BitFieldInsertInstruction : public InstructionBase_4<T, T, T, UInt32Type, UInt32Type>
{
public:
	REQUIRE_TYPE_PARAM(BitFieldInsertInstruction,
		REQUIRE_EXACT(T,
			Bit16Type, Bit32Type
		)
	);

	using InstructionBase_4<T, T, T, UInt32Type, UInt32Type>::InstructionBase;

	static std::string Mnemonic() { return "bfi"; }

	std::string OpCode() const override
	{
		return Mnemonic() + T::Name();
	}
};

}
