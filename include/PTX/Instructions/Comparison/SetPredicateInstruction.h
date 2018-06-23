#pragma once

#include <sstream>

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/PredicateModifier.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Extended/DualOperand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T, bool Typecheck = true>
class SetPredicateInstruction : public PredicatedInstruction, public FlushSubnormalModifier<T>, public PredicateModifier
{
public:
	REQUIRE_TYPE(SetPredicateInstruction,
		Bit16Type, Bit32Type, Bit64Type,
		Int16Type, Int32Type, Int64Type,
		UInt16Type, UInt32Type, UInt64Type,
		Float16Type, Float16x2Type, Float32Type, Float64Type
	);

	SetPredicateInstruction(const Register<PredicateType> *destination, const TypedOperand<T> *sourceA, const TypedOperand<T> *sourceB, typename T::ComparisonOperator comparator) : m_destinationP(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_comparator(comparator) {}

	SetPredicateInstruction(const Register<PredicateType> *destinationP, const Register<PredicateType> *destinationQ, const TypedOperand<T> *sourceA, const TypedOperand<T> *sourceB, typename T::ComparisonOperator comparator, const Register<PredicateType> *sourceC, BoolOperator boolOperator, bool negateSourcePredicate = false) : m_destinationP(destinationP), m_destinationQ(destinationQ), m_sourceA(sourceA), m_sourceB(sourceB), m_comparator(comparator), PredicateModifier(sourceC, boolOperator, negateSourcePredicate) {}

	std::string OpCode() const override
	{
		std::ostringstream code;
		code << "setp" << T::ComparisonOperatorString(m_comparator) + PredicateModifier::OpCodeModifier();
		if constexpr(FlushSubnormalModifier<T>::Enabled)
		{
			code << FlushSubnormalModifier<T>::OpCodeModifier();
		}
		code << T::Name();
		return code.str();
	}

	std::vector<const Operand *> Operands() const override
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
		const Operand *modifier = PredicateModifier::OperandsModifier();
		if (modifier != nullptr)
		{
			operands.push_back(modifier);
		}
		return operands;
	}

private:
	const Register<PredicateType> *m_destinationP = nullptr;
	const Register<PredicateType> *m_destinationQ = nullptr;
	const TypedOperand<T> *m_sourceA = nullptr;
	const TypedOperand<T> *m_sourceB = nullptr;
	typename T::ComparisonOperator m_comparator;
};

template<>
class SetPredicateInstruction<Float16Type> : public InstructionBase_2<PredicateType, Float16Type>, public FlushSubnormalModifier<Float16Type>, public PredicateModifier
{
public:
	SetPredicateInstruction(const Register<PredicateType> *destination, const TypedOperand<Float16Type> *sourceA, const TypedOperand<Float16Type> *sourceB, Float16Type::ComparisonOperator comparator) : InstructionBase_2<PredicateType, Float16Type>(destination, sourceA, sourceB), m_comparator(comparator) {}

	SetPredicateInstruction(const Register<PredicateType> *destination, const TypedOperand<Float16Type> *sourceA, const TypedOperand<Float16Type> *sourceB, Float16Type::ComparisonOperator comparator, const Register<PredicateType> *sourceC, BoolOperator boolOperator, bool negateSourcePredicate = false) : InstructionBase_2<PredicateType, Float16Type>(destination, sourceA, sourceB), m_comparator(comparator), PredicateModifier(sourceC, boolOperator, negateSourcePredicate) {}

	std::string OpCode() const override
	{
		return "setp" + Float16Type::ComparisonOperatorString(m_comparator) + PredicateModifier::OpCodeModifier() + FlushSubnormalModifier<Float16Type>::OpCodeModifier() + Float16Type::Name();
	}

	std::vector<const Operand *> Operands() const override
	{
		auto operands = InstructionBase_2<PredicateType, Float16Type>::Operands();
		const Operand *modifier = PredicateModifier::OperandsModifier();
		if (modifier != nullptr)
		{
			operands.push_back(modifier);
		}
		return operands;
	}

private:
	Float16Type::ComparisonOperator m_comparator;
};

}
