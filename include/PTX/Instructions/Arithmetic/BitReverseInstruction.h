#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Assert = true>
class BitReverseInstruction : public InstructionBase_2<T>
{
public:
	REQUIRE_TYPE_PARAM(BitReverseInstruction,
		REQUIRE_EXACT(T,
			Bit32Type, Bit64Type
		)
	);

	using InstructionBase_2<T>::InstructionBase;

	std::string OpCode() const override
	{
		return "brev" + T::Name();
	}
};

}
