#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class BitFindInstruction : public InstructionBase<T, 1, UInt32Type>
{
	REQUIRE_BASE_TYPE(BitFindInstruction, ScalarType);
	DISABLE_EXACT_TYPE(BitFindInstruction, Int8Type);
	DISABLE_EXACT_TYPE(BitFindInstruction, UInt8Type);
	DISABLE_EXACT_TYPE(BitFindInstruction, Int16Type);
	DISABLE_EXACT_TYPE(BitFindInstruction, UInt16Type);
	DISABLE_EXACT_TYPE_TEMPLATE(BitFindInstruction, BitType);
	DISABLE_EXACT_TYPE_TEMPLATE(BitFindInstruction, FloatType);
public:
	using InstructionBase<T, 1, UInt32Type>::InstructionBase;

	bool GetShiftAmount() const { return m_shiftAmount; }
	void SetShiftAmount(bool shiftAmount) { m_shiftAmount; }

	std::string OpCode() const
	{
		if (m_shiftAmount)
		{
			return "bfind.shiftamt" + T::Name();
		}
		return "bfind" + T::Name();
	}

private:
	bool m_shiftAmount = false;
};

}
