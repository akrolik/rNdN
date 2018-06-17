#pragma once

#include <sstream>

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Typecheck = true>
class SelectInstruction : public InstructionBase_3<T, T, T, PredicateType>
{
public:
	REQUIRE_TYPE(SelectInstruction,
		Bit16Type, Bit32Type, Bit64Type,
		Int16Type, Int32Type, Int64Type,
		UInt16Type, UInt32Type, UInt64Type,
		Float32Type, Float64Type
	);

	using InstructionBase_3<T, T, T, PredicateType>::InstructionBase_3;

	std::string OpCode() const override
	{
		return "selp" + T::Name();
	}
};

}
