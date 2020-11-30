#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Type.h"
#include "PTX/Tree/Operands/Operand.h"
#include "PTX/Tree/Operands/Extended/DualOperand.h"
#include "PTX/Tree/Operands/Extended/HexOperand.h"
#include "PTX/Tree/Operands/Variables/Register.h"

namespace PTX {

template<class T, bool Assert = true>
class MatchAllInstruction : public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(MatchInstruction,
		REQUIRE_EXACT(T, Bit32Type, Bit64Type)
	);

	MatchAllInstruction(const Register<Bit32Type> *destination, const TypedOperand<T> *source, uint32_t memberMask) : MatchAllInstruction(destination, nullptr, source, memberMask) {}
	MatchAllInstruction(const Register<Bit32Type> *destinationD, const Register<PredicateType> *destinationP, const TypedOperand<T> *source, uint32_t memberMask) : m_destinationD(destinationD), m_destinationP(destinationP), m_source(source), m_memberMask(memberMask) {}

	const Register<Bit32Type> *GetDestinationD() const { return m_destinationD; }
	void SetDestinationD(const Register<Bit32Type> *destination) { m_destinationD = destination; }

	const Register<PredicateType> *GetDestinationP() const { return m_destinationP; }
	void SetDestinationP(const Register<PredicateType> *destination) { m_destinationP = destination; }

	const TypedOperand<T> *GetSource() const { return m_source; }
	void SetSource(const TypedOperand<T> *source) { m_source = source; }

	uint32_t GetMemberMask() const { return m_memberMask; }
	void SetMemberMask(uint32_t memberMask) { m_memberMask = memberMask; }

	static std::string Mnemonic() { return "match.all.sync"; }

	std::string OpCode() const override
	{
		return Mnemonic() + T::Name();
	}

	std::vector<const Operand *> Operands() const override
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
		operands.push_back(new HexOperand(m_memberMask));
		return operands;
	}

protected:
	const Register<Bit32Type> *m_destinationD = nullptr;
	const Register<PredicateType> *m_destinationP = nullptr;
	const TypedOperand<T> *m_source = nullptr;
	uint32_t m_memberMask = 0;
};

template<class T, bool Assert = true>
class MatchAnyInstruction : public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(MatchInstruction,
		REQUIRE_EXACT(T, Bit32Type, Bit64Type)
	);

	MatchAnyInstruction(const Register<Bit32Type> *destination, const TypedOperand<T> *source, uint32_t memberMask) : m_destination(destination), m_source(source), m_memberMask(memberMask) {}

	const Register<Bit32Type> *GetDestination() const { return m_destination; }
	void SetDestination(const Register<Bit32Type> *destination) { m_destination = destination; }

	const TypedOperand<T> *GetSource() const { return m_source; }
	void SetSource(const TypedOperand<T> *source) { m_source = source; }

	uint32_t GetMemberMask() const { return m_memberMask; }
	void SetMemberMask(uint32_t memberMask) { m_memberMask = memberMask; }

	static std::string Mnemonic() { return "match.any.sync"; }

	std::string OpCode() const override
	{
		return Mnemonic() + T::Name();
	}

	std::vector<const Operand *> Operands() const override
	{
		std::vector<const Operand *> operands;
		operands.push_back(m_destination);
		operands.push_back(m_source);
		operands.push_back(new HexOperand(m_memberMask));
		return operands;
	}

protected:
	const Register<Bit32Type> *m_destination = nullptr;
	const TypedOperand<T> *m_source = nullptr;
	uint32_t m_memberMask = 0;
};

}
