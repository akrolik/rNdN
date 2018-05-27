#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class BitFieldExtractInstruction : public InstructionBase_3<T, T, UInt32Type, UInt32Type>
{
	REQUIRE_BASE_TYPE(BitFieldExtractInstruction, ScalarType);
	DISABLE_EXACT_TYPE(BitFieldExtractInstruction, Int8Type);
	DISABLE_EXACT_TYPE(BitFieldExtractInstruction, UInt8Type);
	DISABLE_EXACT_TYPE(BitFieldExtractInstruction, Int16Type);
	DISABLE_EXACT_TYPE(BitFieldExtractInstruction, UInt16Type);
	DISABLE_EXACT_TYPE_TEMPLATE(BitFieldExtractInstruction, BitType);
	DISABLE_EXACT_TYPE_TEMPLATE(BitFieldExtractInstruction, FloatType);
public:
	using InstructionBase_3<T, T, UInt32Type, UInt32Type>::InstructionBase;

	std::string OpCode() const
	{
		return "bfe" + T::Name();
	}
};

}
