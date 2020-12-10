#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Type.h"
#include "PTX/Tree/Operands/Operand.h"
#include "PTX/Tree/Operands/Variables/Register.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface(BarrierReductionInstruction)

template<class T>
class BarrierReductionInstructionBase : DispatchInherit(BarrierReductionInstruction), public PredicatedInstruction
{
public:
	BarrierReductionInstructionBase(Register<T> *destination, TypedOperand<UInt32Type> *barrier, Register<PredicateType> *sourcePredicate, bool negateSourcePredicate = false, bool aligned = false) : BarrierReductionInstructionBase(destination, barrier, nullptr, sourcePredicate, negateSourcePredicate, aligned) {}
	BarrierReductionInstructionBase(Register<T> *destination, TypedOperand<UInt32Type> *barrier, TypedOperand<UInt32Type> *threads, Register<PredicateType> *sourcePredicate, bool negateSourcePredicate = false, bool aligned = false) : m_destination(destination), m_barrier(barrier), m_threads(threads), m_sourcePredicate(sourcePredicate), m_negateSourcePredicate(negateSourcePredicate), m_aligned(aligned) {}

	// Properties

	const Register<T> *GetDestination() const { return m_destination; }
	Register<T> *GetDestination() { return m_destination; }
	void SetDestination(Register<T> *destination) { m_destination = destination; }

	const TypedOperand<UInt32Type> *GetBarrier() const { return m_barrier; }
	TypedOperand<UInt32Type> *GetBarrier() { return m_barrier; }
	void SetBarrier(TypedOperand<UInt32Type> *barrier) { m_barrier = barrier; }

	const TypedOperand<UInt32Type> *GetThreads() const { return m_threads; }
	TypedOperand<UInt32Type> *GetThreads() { return m_threads; }
	void SetThreads(TypedOperand<UInt32Type> *threads) { m_threads = threads; }

	const Register<PredicateType> *GetSourcePredicate() const { return m_sourcePredicate; }
	Register<PredicateType> *GetSourcePredicate() { return m_sourcePredicate; }
	void SetSourcePredicate(Register<PredicateType> *source) { m_sourcePredicate = source; }

	bool GetNegateSourcePredicate() const { return m_negateSourcePredicate; }
	void SetNegateSourcePredicate(bool negateSourcePredicate) { m_negateSourcePredicate = negateSourcePredicate; }

	bool GetAligned() { return m_aligned; }
	void SetAligned(bool aligned) { m_aligned = aligned; }

	// Formatting

	std::vector<const Operand *> GetOperands() const override
	{
		std::vector<const Operand *> operands;
		operands.push_back(m_destination);
		operands.push_back(m_barrier);
		if (m_threads != nullptr)
		{
			operands.push_back(m_threads);
		}
		if (m_negateSourcePredicate)
		{
			operands.push_back(new InvertedOperand(m_sourcePredicate));
		}
		else
		{
			operands.push_back(m_sourcePredicate);
		}
		return operands;
	}

	std::vector<Operand *> GetOperands() override
	{
		std::vector<Operand *> operands;
		operands.push_back(m_destination);
		operands.push_back(m_barrier);
		if (m_threads != nullptr)
		{
			operands.push_back(m_threads);
		}
		if (m_negateSourcePredicate)
		{
			operands.push_back(new InvertedOperand(m_sourcePredicate));
		}
		else
		{
			operands.push_back(m_sourcePredicate);
		}
		return operands;
	}
	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:                
	DispatchMember_Type(T);

	Register<T> *m_destination = nullptr;
	TypedOperand<UInt32Type> *m_barrier = nullptr;
	TypedOperand<UInt32Type> *m_threads = nullptr;
	Register<PredicateType> *m_sourcePredicate;
	bool m_negateSourcePredicate = false;
	bool m_aligned = false;
};

template<class T, bool Assert = true>
class BarrierReductionInstruction : public BarrierReductionInstructionBase<T>
{
public:
	REQUIRE_TYPE_PARAM(BarrierReductionInstruction,
		REQUIRE_EXACT(T, PredicateType, UInt32Type)
	);

	// Formatting

	static std::string Mnemonic() { return "barrier.red"; }
};

template<>
class BarrierReductionInstruction<UInt32Type> : public BarrierReductionInstructionBase<UInt32Type>
{
public:
	using BarrierReductionInstructionBase<UInt32Type>::BarrierReductionInstructionBase;

	// formatting

	static std::string Mnemonic() { return "barrier.red"; }
                      
	std::string GetOpCode() const override
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

	BarrierReductionInstruction(Register<PredicateType> *destination, TypedOperand<UInt32Type> *barrier, Operation operation, Register<PredicateType> *sourcePredicate, bool negateSourcePredicate = false, bool aligned = false) : BarrierReductionInstruction<PredicateType>(destination, barrier, nullptr, operation, sourcePredicate, negateSourcePredicate, aligned) {}
	BarrierReductionInstruction(Register<PredicateType> *destination, TypedOperand<UInt32Type> *barrier, TypedOperand<UInt32Type> *threads, Operation operation, Register<PredicateType> *sourcePredicate, bool negateSourcePredicate = false, bool aligned = false) : BarrierReductionInstructionBase<PredicateType>(destination, barrier, threads, sourcePredicate, negateSourcePredicate, aligned), m_operation(operation) {}

	// Properties

	Operation GetOperation() const { return m_operation; }
	void SetOperation(Operation operation) { m_operation = operation; }

	// Formatting

	static std::string Mnemonic() { return "barrier.red"; }

	std::string GetOpCode() const override
	{
		std::string code = Mnemonic() + OperationString(m_operation);
		if (this->m_aligned)
		{
			code += ".aligned";
		}
		return code + PredicateType::Name();
	}

protected:
	Operation m_operation;
};

DispatchImplementation(BarrierReductionInstruction)

}
