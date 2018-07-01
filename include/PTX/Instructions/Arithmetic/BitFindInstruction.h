#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Assert = true>
class BitFindInstruction : public InstructionBase_1<UInt32Type, T>
{
public:
	REQUIRE_TYPE_PARAM(BitFindInstruction,
		REQUIRE_EXACT(T,
			Int32Type, Int64Type,
			UInt32Type, UInt64Type
		)
	);

	using InstructionBase_1<UInt32Type, T>::InstructionBase;

	bool GetShiftAmount() const { return m_shiftAmount; }
	void SetShiftAmount(bool shiftAmount) { m_shiftAmount; }

	std::string OpCode() const override
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
