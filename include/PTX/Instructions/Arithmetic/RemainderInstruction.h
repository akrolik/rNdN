#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Assert = true>
class RemainderInstruction : public InstructionBase_2<T>
{
public:
	REQUIRE_TYPE_PARAM(RemainderInstruction,
		REQUIRE_EXACT(T,
			Int16Type, Int32Type, Int64Type,
			UInt16Type, UInt32Type, UInt64Type
		)
	);

	using InstructionBase_2<T>::InstructionBase_2;

	std::string OpCode() const override
	{
		return "rem" + T::Name();
	}
};

}
