#pragma once

#include <sstream>

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/PredicateModifier.h"

#include "PTX/Operands/Operand.h"
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

	SetPredicateInstruction(const Register<PredicateType> *destination, const Operand<T> *sourceA, const Operand<T> *sourceB, typename T::ComparisonOperator comparator) : m_destinationP(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_comparator(comparator) {}

	SetPredicateInstruction(const Register<PredicateType> *destinationP, const Register<PredicateType> *destinationQ, const Operand<T> *sourceA, const Operand<T> *sourceB, typename T::ComparisonOperator comparator, const Register<PredicateType> *sourceC, BoolOperator boolOperator, bool negateSourcePredicate = false) : m_destinationP(destinationP), m_destinationQ(destinationQ), m_sourceA(sourceA), m_sourceB(sourceB), m_comparator(comparator), PredicateModifier(sourceC, boolOperator, negateSourcePredicate) {}

	std::string OpCode() const override
	{
		std::ostringstream code;
		code << "setp" << T::ComparisonOperatorString(m_comparator) + PredicateModifier::OpCodeModifier();
		if constexpr(T::FlushModifier)
		{
			code << FlushSubnormalModifier<T>::OpCodeModifier();
		}
		code << T::Name();
		return code.str();
	}

	std::string Operands() const override
	{
		std::ostringstream code;
		code << m_destinationP->ToString();
		if (m_destinationQ != nullptr)
		{
			code << "|" << m_destinationQ->ToString();
		}
		code << ", " << m_sourceA->ToString() << ", " << m_sourceB->ToString() << PredicateModifier::OperandsModifier();
		return code.str();
	}

private:
	const Register<PredicateType> *m_destinationP = nullptr;
	const Register<PredicateType> *m_destinationQ = nullptr;
	const Operand<T> *m_sourceA = nullptr;
	const Operand<T> *m_sourceB = nullptr;
	typename T::ComparisonOperator m_comparator;
};

template<>
class SetPredicateInstruction<Float16Type> : public InstructionBase_2<PredicateType, Float16Type>, public FlushSubnormalModifier<Float16Type>, public PredicateModifier
{
public:
	SetPredicateInstruction(const Register<PredicateType> *destination, const Operand<Float16Type> *sourceA, const Operand<Float16Type> *sourceB, Float16Type::ComparisonOperator comparator) : InstructionBase_2<PredicateType, Float16Type>(destination, sourceA, sourceB), m_comparator(comparator) {}

	SetPredicateInstruction(const Register<PredicateType> *destination, const Operand<Float16Type> *sourceA, const Operand<Float16Type> *sourceB, Float16Type::ComparisonOperator comparator, const Register<PredicateType> *sourceC, BoolOperator boolOperator, bool negateSourcePredicate = false) : InstructionBase_2<PredicateType, Float16Type>(destination, sourceA, sourceB), m_comparator(comparator), PredicateModifier(sourceC, boolOperator, negateSourcePredicate) {}

	std::string OpCode() const override
	{
		return "setp" + Float16Type::ComparisonOperatorString(m_comparator) + PredicateModifier::OpCodeModifier() + FlushSubnormalModifier<Float16Type>::OpCodeModifier() + Float16Type::Name();
	}

	std::string Operands() const override
	{
		return InstructionBase_2<PredicateType, Float16Type>::Operands() + PredicateModifier::OperandsModifier();
	}

private:
	Float16Type::ComparisonOperator m_comparator;
};

}
