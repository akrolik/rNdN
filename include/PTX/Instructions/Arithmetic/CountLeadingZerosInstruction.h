#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Assert = true>
class CountLeadingZerosInstruction : public InstructionBase_1<UInt32Type, T>
{
public:
	REQUIRE_TYPE_PARAM(CountLeadingZerosInstruction,
		REQUIRE_EXACT(T,
			Bit32Type, Bit64Type
		)
	);

	using InstructionBase_1<UInt32Type, T>::InstructionBase;

	std::string OpCode() const override
	{
		return "clz" + T::Name();
	}
};

}
