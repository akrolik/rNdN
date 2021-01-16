#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

DispatchInterface(BitFindInstruction)

template<class T, bool Assert = true>
class BitFindInstruction : DispatchInherit(BitFindInstruction), public InstructionBase_1<UInt32Type, T>
{
public:
	REQUIRE_TYPE_PARAM(BitFindInstruction,
		REQUIRE_EXACT(T,
			Int32Type, Int64Type,
			UInt32Type, UInt64Type
		)
	);

	BitFindInstruction(Register<UInt32Type> *destination, TypedOperand<T> *source, bool shiftAmount = false) : InstructionBase_1<UInt32Type, T>(destination, source), m_shiftAmount(shiftAmount) {}

	// Properties

	bool GetShiftAmount() const { return m_shiftAmount; }
	void SetShiftAmount(bool shiftAmount) { m_shiftAmount; }

	// Formatting

	static std::string Mnemonic() { return "bfind"; }

	std::string GetOpCode() const override
	{
		std::string code = Mnemonic();
		if (m_shiftAmount)
		{
			code += ".shiftamt";
		}
		return code + T::Name();
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);

	bool m_shiftAmount = false;
};

}
