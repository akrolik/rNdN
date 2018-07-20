#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Type.h"
#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class BarrierReductionInstructionBase : public PredicatedInstruction
{
public:
	BarrierReductionInstructionBase(const Register<T> *destination, const TypedOperand<UInt32Type> *barrier, const Register<PredicateType> *predicate, bool negatePredicate = false, bool aligned = false) : BarrierReductionInstructionBase(destination, barrier, nullptr, predicate, negatePredicate, aligned) {}
	BarrierReductionInstructionBase(const Register<T> *destination, const TypedOperand<UInt32Type> *barrier, const TypedOperand<UInt32Type> *threads, const Register<PredicateType> *predicate, bool negatePredicate = false, bool aligned = false) : m_destination(destination), m_barrier(barrier), m_threads(threads), m_predicate(predicate), m_negatePredicate(negatePredicate), m_aligned(aligned) {}

	bool GetAligned() { return m_aligned; }
	void SetAligned(bool aligned) { m_aligned = aligned; }

	std::vector<const Operand *> Operands() const override
	{
		std::vector<const Operand *> operands;
		operands.push_back(m_destination);
		operands.push_back(m_barrier);
		if (m_threads != nullptr)
		{
			operands.push_back(m_threads);
		}
		if (m_negatePredicate)
		{
			operands.push_back(new InvertedOperand(m_predicate));
		}
		else
		{
			operands.push_back(m_predicate);
		}
		return operands;
	}

protected:                
	const Register<T> *m_destination = nullptr;
	const TypedOperand<UInt32Type> *m_barrier = nullptr;
	const TypedOperand<UInt32Type> *m_threads = nullptr;
	const Register<PredicateType> *m_predicate;
	bool m_negatePredicate = false;
	bool m_aligned = false;
};

template<class T>
class BarrierReductionInstruction : public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(AbsoluteInstruction,
		REQUIRE_EXACT(T, PredicateType, UInt32Type)
	);

	static std::string Mnemonic() { return "barrier.red"; }
};

template<>
class BarrierReductionInstruction<UInt32Type> : public BarrierReductionInstructionBase<UInt32Type>
{
public:
	using BarrierReductionInstructionBase<UInt32Type>::BarrierReductionInstructionBase;

	static std::string Mnemonic() { return "barrier.red"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic() + ".popc";
		if (this->m_aligned)
		{
			code += ".aligned";
		}
		return code + UInt32Type::Name();
	}
};

template<>
class BarrierReductionInstruction<PredicateType> : public BarrierReductionInstructionBase<PredicateType>
{
public:
	enum class Operation {
		And,
		Or
	};

	static std::string OperationString(Operation operation)
	{
		switch (operation)
		{
			case Operation::And:
				return ".and";
			case Operation::Or:
				return ".or";
		}
		return ".<unknown>";
	}

	BarrierReductionInstruction(const Register<PredicateType> *destination, const TypedOperand<UInt32Type> *barrier, Operation operation, const Register<PredicateType> *predicate, bool negatePredicate = false, bool aligned = false) : BarrierReductionInstruction<PredicateType>(destination, barrier, nullptr, operation, predicate, negatePredicate, aligned) {}
	BarrierReductionInstruction(const Register<PredicateType> *destination, const TypedOperand<UInt32Type> *barrier, const TypedOperand<UInt32Type> *threads, Operation operation, const Register<PredicateType> *predicate, bool negatePredicate = false, bool aligned = false) : BarrierReductionInstructionBase<PredicateType>(destination, barrier, threads, predicate, negatePredicate, aligned), m_operation(operation) {}

	static std::string Mnemonic() { return "barrier.red"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic() + OperationString(m_operation);
		if (this->m_aligned)
		{
			code += ".aligned";
		}
		return code + PredicateType::Name();
	}

private:
	Operation m_operation;
};

}
