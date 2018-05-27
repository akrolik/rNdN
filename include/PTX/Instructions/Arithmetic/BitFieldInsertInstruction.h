#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class BitFieldInsertInstruction : public InstructionBase_4<T, T, T, UInt32Type, UInt32Type>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(BitFieldInsertInstruction, BitType);
	DISABLE_EXACT_TYPE(BitFieldInsertInstruction, PredicateType);
	DISABLE_EXACT_TYPE(BitFieldInsertInstruction, Bit8Type);
	DISABLE_EXACT_TYPE(BitFieldInsertInstruction, Bit16Type);
public:
	using InstructionBase_4<T, T, T, UInt32Type, UInt32Type>::InstructionBase;

	std::string OpCode() const
	{
		return "bfi" + T::Name();
	}
};

}
