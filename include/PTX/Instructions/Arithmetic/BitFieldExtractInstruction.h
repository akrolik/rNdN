#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Assert = true>
class BitFieldExtractInstruction : public InstructionBase_3<T, T, UInt32Type, UInt32Type>
{
public:
	REQUIRE_TYPE_PARAM(BitFieldExtractInstruction,
		REQUIRE_EXACT(T,
			Int32Type, Int64Type,
			UInt32Type, UInt64Type
		)
	);

	using InstructionBase_3<T, T, UInt32Type, UInt32Type>::InstructionBase;

	std::string OpCode() const override
	{
		return "bfe" + T::Name();
	}
};

}
