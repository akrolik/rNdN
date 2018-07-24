#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Type.h"
#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Extended/HexOperand.h"
#include "PTX/Operands/Extended/InvertedOperand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class VoteInstructionBase : public PredicatedInstruction
{
public:
	VoteInstructionBase(const Register<T> *destination, const TypedOperand<PredicateType> *sourcePredicate, uint32_t memberMask, bool negateSourcePredicate = false) : m_destination(destination), m_sourcePredicate(sourcePredicate), m_negateSourcePredicate(negateSourcePredicate), m_memberMask(memberMask) {}

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

	static std::string Mnemonic() { return "vote.sync"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ModeString(m_mode) + PredicateType::Name();
	}

private:
	Mode m_mode;
};

}
