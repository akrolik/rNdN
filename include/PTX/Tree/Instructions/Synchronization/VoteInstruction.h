#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Type.h"
#include "PTX/Tree/Operands/Operand.h"
#include "PTX/Tree/Operands/Extended/HexOperand.h"
#include "PTX/Tree/Operands/Extended/InvertedOperand.h"
#include "PTX/Tree/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class VoteInstructionBase : public PredicatedInstruction
{
public:
	VoteInstructionBase(const Register<T> *destination, const TypedOperand<PredicateType> *sourcePredicate, uint32_t memberMask, bool negateSourcePredicate = false) : m_destination(destination), m_sourcePredicate(sourcePredicate), m_negateSourcePredicate(negateSourcePredicate), m_memberMask(memberMask) {}

	const Register<T> *GetDestination() const { return m_destination; }
	void SetDestination(const Register<T> *destination) { m_destination = destination; }

	const TypedOperand<PredicateType> *GetSourcePredicate() const { return m_sourcePredicate; }
	void SetSourcePredicate(const TypedOperand<PredicateType> *source) { m_sourcePredicate = source; }

	bool GetNegateSourcePredicate() const { return m_negateSourcePredicate; }
	void SetNegateSourcePredicate(bool negateSourcePredicate) { m_negateSourcePredicate = negateSourcePredicate; }

	uint32_t GetMemberMask() const { return m_memberMask; }
	void SetMemberMask(uint32_t memberMask) { m_memberMask = memberMask; }

	std::vector<const Operand *> Operands() const override
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
		operands.push_back(new HexOperand(m_memberMask));
		return operands;
	}

protected:                
	const Register<T> *m_destination = nullptr;
	const TypedOperand<PredicateType> *m_sourcePredicate = nullptr;
	bool m_negateSourcePredicate = false;
	uint32_t m_memberMask = 0;
};

template<class T, bool Assert = true>
class VoteInstruction : public PredicatedInstruction
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

	static std::string Mnemonic() { return "vote.sync"; }

	std::string OpCode() const override
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

	VoteInstruction(const Register<PredicateType> *destination, const TypedOperand<PredicateType> *sourcePredicate, uint32_t memberMask, Mode mode, bool negateSourcePredicate = false) : VoteInstructionBase<PredicateType>(destination, sourcePredicate, negateSourcePredicate, memberMask), m_mode(mode) {}

	Mode GetMode() const { return m_mode; }
	void SetMode(Mode mode) { m_mode = mode ;}

	static std::string Mnemonic() { return "vote.sync"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ModeString(m_mode) + PredicateType::Name();
	}

private:
	Mode m_mode;
};

}
