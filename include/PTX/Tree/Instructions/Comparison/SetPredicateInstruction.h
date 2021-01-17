#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/PredicatedInstruction.h"
#include "PTX/Tree/Instructions/Modifiers/ComparisonModifier.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Tree/Instructions/Modifiers/PredicateModifier.h"

#include "PTX/Tree/Type.h"
#include "PTX/Tree/Operands/Operand.h"
#include "PTX/Tree/Operands/Extended/DualOperand.h"
#include "PTX/Tree/Operands/Variables/Registers/Register.h"

namespace PTX {

DispatchInterface(SetPredicateInstruction)

template<class T, bool Assert = true>
class SetPredicateInstruction : DispatchInherit(SetPredicateInstruction), public PredicatedInstruction, public ComparisonModifier<T>, public FlushSubnormalModifier<T>, public PredicateModifier
{
public:
	REQUIRE_TYPE_PARAM(SetPredicateInstruction,
		REQUIRE_EXACT(T,
			Bit16Type, Bit32Type, Bit64Type,
			Int16Type, Int32Type, Int64Type,
			UInt16Type, UInt32Type, UInt64Type,
			Float16Type, Float16x2Type, Float32Type, Float64Type
		)
	);

	SetPredicateInstruction(Register<PredicateType> *destination, TypedOperand<T> *sourceA, TypedOperand<T> *sourceB, typename T::ComparisonOperator comparator) : ComparisonModifier<T>(comparator), m_destinationP(destination), m_sourceA(sourceA), m_sourceB(sourceB) {}

	SetPredicateInstruction(Register<PredicateType> *destinationP, Register<PredicateType> *destinationQ, TypedOperand<T> *sourceA, TypedOperand<T> *sourceB, typename T::ComparisonOperator comparator, Register<PredicateType> *sourceC, BoolOperator boolOperator, bool negateSourcePredicate = false) : ComparisonModifier<T>(comparator), m_destinationP(destinationP), m_destinationQ(destinationQ), m_sourceA(sourceA), m_sourceB(sourceB), PredicateModifier(sourceC, boolOperator, negateSourcePredicate) {}

	// Properties

	const Register<PredicateType> *GetDestination() const { return m_destinationP; }
	Register<PredicateType> *GetDestination() { return m_destinationP; }
	void SetDestination(Register<PredicateType> *destination) { m_destinationP = destination; }

	const Register<PredicateType> *GetDestinationQ() const { return m_destinationQ; }
	Register<PredicateType> *GetDestinationQ() { return m_destinationQ; }
	void SetDestinationQ(Register<PredicateType> *destination) { m_destinationQ = destination; }

	const TypedOperand<T> *GetSourceA() const { return m_sourceA; }
	TypedOperand<T> *GetSourceA() { return m_sourceA; }
	void SetSourceA(TypedOperand<T> *source) { m_sourceA = source; }

	const TypedOperand<T> *GetSourceB() const { return m_sourceB; }
	TypedOperand<T> *GetSourceB() { return m_sourceB; }
	void SetSourceB(TypedOperand<T> *source) { m_sourceB = source; }

	// Formatting

	static std::string Mnemonic() { return "setp"; }

	std::string GetOpCode() const override
	{
		std::string code = Mnemonic() + ComparisonModifier<T>::GetOpCodeModifier() + PredicateModifier::GetOpCodeModifier();
		if constexpr(FlushSubnormalModifier<T>::Enabled)
		{
			code += FlushSubnormalModifier<T>::GetOpCodeModifier();
		}
		return code + T::Name();
	}

	std::vector<const Operand *> GetOperands() const override
	{
		std::vector<const Operand *> operands;
		if (m_destinationQ == nullptr)
		{
			operands.push_back(m_destinationP);
		}
		else
		{
			operands.push_back(new DualOperand(m_destinationP, m_destinationQ));
		}
		operands.push_back(m_sourceA);
		operands.push_back(m_sourceB);
		if (const auto modifier = PredicateModifier::GetOperandsModifier())
		{
			operands.push_back(modifier);
		}
		return operands;
	}

	std::vector<Operand *> GetOperands() override
	{
		std::vector<Operand *> operands;
		if (m_destinationQ == nullptr)
		{
			operands.push_back(m_destinationP);
		}
		else
		{
			operands.push_back(new DualOperand(m_destinationP, m_destinationQ));
		}
		operands.push_back(m_sourceA);
		operands.push_back(m_sourceB);
		if (auto modifier = PredicateModifier::GetOperandsModifier())
		{
			operands.push_back(modifier);
		}
		return operands;
	}

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);

	Register<PredicateType> *m_destinationP = nullptr;
	Register<PredicateType> *m_destinationQ = nullptr;
	TypedOperand<T> *m_sourceA = nullptr;
	TypedOperand<T> *m_sourceB = nullptr;
};

template<>
class SetPredicateInstruction<Float16Type> : DispatchInherit(SetPredicateInstruction), public InstructionBase_2<PredicateType, Float16Type>, public ComparisonModifier<Float16Type>, public FlushSubnormalModifier<Float16Type>, public PredicateModifier
{
public:
	SetPredicateInstruction(Register<PredicateType> *destination, TypedOperand<Float16Type> *sourceA, TypedOperand<Float16Type> *sourceB, Float16Type::ComparisonOperator comparator) : InstructionBase_2<PredicateType, Float16Type>(destination, sourceA, sourceB), ComparisonModifier<Float16Type>(comparator) {}

	SetPredicateInstruction(Register<PredicateType> *destination, TypedOperand<Float16Type> *sourceA, TypedOperand<Float16Type> *sourceB, Float16Type::ComparisonOperator comparator, Register<PredicateType> *sourceC, BoolOperator boolOperator, bool negateSourcePredicate = false) : InstructionBase_2<PredicateType, Float16Type>(destination, sourceA, sourceB), ComparisonModifier<Float16Type>(comparator), PredicateModifier(sourceC, boolOperator, negateSourcePredicate) {}

	static std::string Mnemonic() { return "setp"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + ComparisonModifier<Float16Type>::GetOpCodeModifier() + PredicateModifier::GetOpCodeModifier() + FlushSubnormalModifier<Float16Type>::GetOpCodeModifier() + Float16Type::Name();
	}

	std::vector<const Operand *> GetOperands() const override
	{
		auto operands = InstructionBase_2<PredicateType, Float16Type>::GetOperands();
		if (const auto modifier = PredicateModifier::GetOperandsModifier())
		{
			operands.push_back(modifier);
		}
		return operands;
	}

	std::vector<Operand *> GetOperands() override
	{
		auto operands = InstructionBase_2<PredicateType, Float16Type>::GetOperands();
		if (auto modifier = PredicateModifier::GetOperandsModifier())
		{
			operands.push_back(modifier);
		}
		return operands;
	}

	// Visitor

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(Float16Type);
};

}
