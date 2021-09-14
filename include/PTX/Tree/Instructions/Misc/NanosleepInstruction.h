#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Type.h"
#include "PTX/Tree/Operands/Operand.h"

namespace PTX {

DispatchInterface(NanosleepInstruction)

template<class T, bool Assert = true>
class NanosleepInstruction : DispatchInherit(ActiveMaskInstruction), public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(NanosleepInstruction,
		REQUIRE_EXACT(T, UInt32Type)
	);

	NanosleepInstruction(TypedOperand<T> *source) : m_source(source) {}
	 
	// Properties

	const TypedOperand<T> *GetSource() const { return m_source; }
	TypedOperand<T> *GetSource() { return m_source; }
	void SetSource(TypedOperand<T> *source) { m_source = source; }

	// Operands

	std::vector<const Operand *> GetOperands() const override
	{
		return { m_source };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { m_source };
	}

	// Analysis properties

	bool HasSideEffect() const override { return true; }

	// Formatting

	static std::string Mnemonic() { return "nanosleep"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + T::Name();
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);

	TypedOperand<T> *m_source = nullptr;
};

}
