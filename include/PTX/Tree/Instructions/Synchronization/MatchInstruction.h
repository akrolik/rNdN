#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Type.h"
#include "PTX/Tree/Operands/Operand.h"
#include "PTX/Tree/Operands/Constants/Value.h"
#include "PTX/Tree/Operands/Extended/DualOperand.h"
#include "PTX/Tree/Operands/Variables/Registers/Register.h"

namespace PTX {

DispatchInterface(MatchAllInstruction)
DispatchInterface(MatchAnyInstruction)

template<class T, bool Assert = true>
class MatchAllInstruction : DispatchInherit(MatchAllInstruction), public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(MatchInstruction,
		REQUIRE_EXACT(T, Bit32Type, Bit64Type)
	);

	MatchAllInstruction(Register<Bit32Type> *destination, TypedOperand<T> *source, UInt32Value *memberMask)
		: MatchAllInstruction(destination, nullptr, source, memberMask) {}

	MatchAllInstruction(Register<Bit32Type> *destinationD, Register<PredicateType> *destinationP, TypedOperand<T> *source, UInt32Value *memberMask)
		: m_destinationD(destinationD), m_destinationP(destinationP), m_source(source), m_memberMask(memberMask) {}

	// Properties

	const Register<Bit32Type> *GetDestinationD() const { return m_destinationD; }
	Register<Bit32Type> *GetDestinationD() { return m_destinationD; }
	void SetDestinationD(Register<Bit32Type> *destination) { m_destinationD = destination; }

	const Register<PredicateType> *GetDestinationP() const { return m_destinationP; }
	Register<PredicateType> *GetDestinationP() { return m_destinationP; }
	void SetDestinationP(Register<PredicateType> *destination) { m_destinationP = destination; }

	const TypedOperand<T> *GetSource() const { return m_source; }
	TypedOperand<T> *GetSource() { return m_source; }
	void SetSource(TypedOperand<T> *source) { m_source = source; }

	const UInt32Value *GetMemberMask() const { return m_memberMask; }
	UInt32Value *GetMemberMask() { return m_memberMask; }
	void SetMemberMask(UInt32Value *memberMask) { m_memberMask = memberMask; }

	// Formatting

	static std::string Mnemonic() { return "match.all.sync"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + T::Name();
	}

	std::vector<const Operand *> GetOperands() const override
	{
		std::vector<const Operand *> operands;
		if (m_destinationP == nullptr)
		{
			operands.push_back(m_destinationD);
		}
		else
		{
			operands.push_back(new DualOperand(m_destinationD, m_destinationP));
		}
		operands.push_back(m_source);
		operands.push_back(m_memberMask);
		return operands;
	}
	
	std::vector<Operand *> GetOperands() override
	{
		std::vector<Operand *> operands;
		if (m_destinationP == nullptr)
		{
			operands.push_back(m_destinationD);
		}
		else
		{
			operands.push_back(new DualOperand(m_destinationD, m_destinationP));
		}
		operands.push_back(m_source);
		operands.push_back(m_memberMask);
		return operands;
	}
	
	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);

	Register<Bit32Type> *m_destinationD = nullptr;
	Register<PredicateType> *m_destinationP = nullptr;
	TypedOperand<T> *m_source = nullptr;
	UInt32Value *m_memberMask = nullptr;
};

template<class T, bool Assert = true>
class MatchAnyInstruction : DispatchInherit(MatchAnyInstruction), public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(MatchInstruction,
		REQUIRE_EXACT(T, Bit32Type, Bit64Type)
	);

	MatchAnyInstruction(Register<Bit32Type> *destination, TypedOperand<T> *source, UInt32Value *memberMask)
		: m_destination(destination), m_source(source), m_memberMask(memberMask) {}

	// Properties

	const Register<Bit32Type> *GetDestination() const { return m_destination; }
	Register<Bit32Type> *GetDestination() { return m_destination; }
	void SetDestination(Register<Bit32Type> *destination) { m_destination = destination; }

	const TypedOperand<T> *GetSource() const { return m_source; }
	TypedOperand<T> *GetSource() { return m_source; }
	void SetSource(TypedOperand<T> *source) { m_source = source; }

	const UInt32Value *GetMemberMask() const { return m_memberMask; }
	UInt32Value *GetMemberMask() { return m_memberMask; }
	void SetMemberMask(UInt32Value *memberMask) { m_memberMask = memberMask; }

	// Formatting

	static std::string Mnemonic() { return "match.any.sync"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + T::Name();
	}

	std::vector<const Operand *> GetOperands() const override
	{
		return { m_destination, m_source, m_memberMask };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { m_destination, m_source, m_memberMask };
	}

	// Visitors
	
	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);

	Register<Bit32Type> *m_destination = nullptr;
	TypedOperand<T> *m_source = nullptr;
	UInt32Value *m_memberMask = nullptr;
};

}
