#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Type.h"
#include "PTX/Tree/Operands/Operand.h"
#include "PTX/Tree/Operands/Constants/Value.h"
#include "PTX/Tree/Operands/Extended/InvertedOperand.h"
#include "PTX/Tree/Operands/Variables/Registers/Register.h"

namespace PTX {

DispatchInterface(VoteInstruction)

template<class T>
class VoteInstructionBase : DispatchInherit(VoteInstruction), public PredicatedInstruction
{
public:
	VoteInstructionBase(Register<T> *destination, TypedOperand<PredicateType> *sourcePredicate, UInt32Value *memberMask, bool negateSourcePredicate = false)
		: m_destination(destination), m_sourcePredicate(sourcePredicate), m_negateSourcePredicate(negateSourcePredicate), m_memberMask(memberMask) {}

	// Analysis properties

	bool HasSideEffect() const override { return true; }

	// Properties

	const Register<T> *GetDestination() const { return m_destination; }
	Register<T> *GetDestination() { return m_destination; }
	void SetDestination(Register<T> *destination) { m_destination = destination; }

	const TypedOperand<PredicateType> *GetSourcePredicate() const { return m_sourcePredicate; }
	TypedOperand<PredicateType> *GetSourcePredicate() { return m_sourcePredicate; }
	void SetSourcePredicate(TypedOperand<PredicateType> *source) { m_sourcePredicate = source; }

	bool GetNegateSourcePredicate() const { return m_negateSourcePredicate; }
	void SetNegateSourcePredicate(bool negateSourcePredicate) { m_negateSourcePredicate = negateSourcePredicate; }

	const UInt32Value *GetMemberMask() const { return m_memberMask; }
	UInt32Value *GetMemberMask() { return m_memberMask; }
	void SetMemberMask(UInt32Value *memberMask) { m_memberMask = memberMask; }
	
	// Formatting

	std::vector<const Operand *> GetOperands() const override
	{
		std::vector<const Operand *> operands;
		operands.push_back(m_destination);
		if (m_negateSourcePredicate)
		{
			operands.push_back(new InvertedOperand(m_sourcePredicate));
		}
		else
		{
			operands.push_back(m_sourcePredicate);
		}
		operands.push_back(m_memberMask);
		return operands;
	}

	std::vector<Operand *> GetOperands() override
	{
		std::vector<Operand *> operands;
		operands.push_back(m_destination);
		if (m_negateSourcePredicate)
		{
			operands.push_back(new InvertedOperand(m_sourcePredicate));
		}
		else
		{
			operands.push_back(m_sourcePredicate);
		}
		operands.push_back(m_memberMask);
		return operands;
	}

	// Visitors
	
	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:                
	DispatchMember_Type(T);

	Register<T> *m_destination = nullptr;
	TypedOperand<PredicateType> *m_sourcePredicate = nullptr;
	bool m_negateSourcePredicate = false;
	UInt32Value *m_memberMask = nullptr;
};

template<class T, bool Assert = true>
class VoteInstruction : public VoteInstructionBase<T>
{
public:
	REQUIRE_TYPE_PARAM(VoteInstruction,
		REQUIRE_EXACT(T, PredicateType, Bit32Type)
	);

	static std::string Mnemonic() { return "vote.sync"; }
};

template<>
class VoteInstruction<Bit32Type> : public VoteInstructionBase<Bit32Type>
{
public:
	using VoteInstructionBase<Bit32Type>::VoteInstructionBase;

	// Formatting

	static std::string Mnemonic() { return "vote.sync"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + ".ballot" + Bit32Type::Name();
	}
};

template<>
class VoteInstruction<PredicateType> : public VoteInstructionBase<PredicateType>
{
public:
	enum class Mode {
		All,
		Any,
		Uniform
	};

	static std::string ModeString(Mode Mode)
	{
		switch (Mode)
		{
			case Mode::All:
				return ".all";
			case Mode::Any:
				return ".any";
			case Mode::Uniform:
				return ".uni";
		}
		return ".<unknown>";
	}

	VoteInstruction(Register<PredicateType> *destination, TypedOperand<PredicateType> *sourcePredicate, UInt32Value *memberMask, Mode mode, bool negateSourcePredicate = false)
		: VoteInstructionBase<PredicateType>(destination, sourcePredicate, memberMask, negateSourcePredicate), m_mode(mode) {}

	// Properties

	Mode GetMode() const { return m_mode; }
	void SetMode(Mode mode) { m_mode = mode ;}

	// Formatting

	static std::string Mnemonic() { return "vote.sync"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + ModeString(m_mode) + PredicateType::Name();
	}

protected:
	Mode m_mode;
};

}
