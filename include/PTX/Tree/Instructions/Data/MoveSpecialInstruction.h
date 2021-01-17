#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Operands/Variables/Registers/Register.h"
#include "PTX/Tree/Operands/Variables/Registers/SpecialRegister.h"

namespace PTX {

DispatchInterface(MoveSpecialInstruction)

template<class T, bool Assert = true>
class MoveSpecialInstruction : DispatchInherit(MoveSpecialInstruction), public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(MoveSpecialInstruction,
		REQUIRE_EXACT(T, Bit32Type, UInt32Type, UInt64Type)
	);

	MoveSpecialInstruction(Register<T> *destination, SpecialRegister<T> *source)
		: m_destination(destination), m_source(source) {}

	// Properties

	const Register<T> *GetDestination() const { return m_destination; }
	Register<T> *GetDestination() { return m_destination; }
	void SetDestination(Register<T> *destination) { m_destination = destination; }

	const SpecialRegister<T> *GetSource() const { return m_source; }
	SpecialRegister<T> *GetSource() { return m_source; }
	void SetSource(SpecialRegister<T> *source) { m_source = source; }

	// Formatting

	static std::string Mnemonic() { return "mov"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + T::Name();
	}

	std::vector<const Operand *> GetOperands() const override
	{
		return { m_destination, m_source };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { m_destination, m_source };
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);

	Register<T> *m_destination = nullptr;
	SpecialRegister<T> *m_source = nullptr;
};

}
