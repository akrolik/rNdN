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

	BitFindInstruction(const Register<UInt32Type> *destination, const TypedOperand<T> *source, bool shiftAmount = false) : InstructionBase_1<UInt32Type, T>(destination, source), m_shiftAmount(shiftAmount) {}

	bool GetShiftAmount() const { return m_shiftAmount; }
	void SetShiftAmount(bool shiftAmount) { m_shiftAmount; }

	static std::string Mnemonic() { return "bfind"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic();
		if (m_shiftAmount)
		{
			code += ".shiftamt";
		}
		return code + T::Name();
	}

private:
	bool m_shiftAmount = false;
};

}
