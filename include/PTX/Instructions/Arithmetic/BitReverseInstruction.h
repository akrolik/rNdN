#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class BitReverseInstruction : public InstructionBase<T, 2>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(BitReverseInstruction, BitType);
	DISABLE_EXACT_TYPE(BitReverseInstruction, PredicateType);
	DISABLE_EXACT_TYPE(BitReverseInstruction, Bit8Type);
	DISABLE_EXACT_TYPE(BitReverseInstruction, Bit16Type);
public:
	using InstructionBase<T, 2>::InstructionBase;

	std::string OpCode() const
	{
		return "brev" + T::Name();
	}
};

}
