#pragma once

#include <sstream>

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T, bool Typecheck = true>
class SetPredicateInstruction : public PredicatedInstruction, public FlushSubnormalModifier<T>
{
public:
	REQUIRE_TYPE(SetPredicateInstruction,
		Bit16Type, Bit32Type, Bit64Type,
		Int16Type, Int32Type, Int64Type,
		UInt16Type, UInt32Type, UInt64Type,
		Float16Type, Float16x2Type, Float32Type, Float64Type
	);

	enum BoolOperator {
		And,
		Or,
		Xor
	};

	SetPredicateInstruction(const Register<PredicateType> *destination, const Operand<T> *sourceA, const Operand<T> *sourceB, typename T::ComparisonOperator comparator) : SetPredicateInstruction(destination, nullptr, sourceA, sourceB, comparator, nullptr, And, false) {}

	SetPredicateInstruction(const Register<PredicateType> *destinationP, const Register<PredicateType> *destinationQ, const Operand<T> *sourceA, const Operand<T> *sourceB, typename T::ComparisonOperator comparator, const Register<PredicateType> *sourceC, BoolOperator boolOperator, bool negateSourcePredicate = false) : m_destinationP(destinationP), m_destinationQ(destinationQ), m_sourceA(sourceA), m_sourceB(sourceB), m_comparator(comparator), m_sourceC(sourceC), m_boolOperator(boolOperator), m_negateSourcePredicate(negateSourcePredicate) {}

	std::string OpCode() const override
	{
		std::ostringstream code;
		code << "setp" << T::ComparisonOperatorString(m_comparator);
		if (m_sourceC != nullptr)
		{
			switch (m_boolOperator)
			{
				case And:
					code << ".and";
					break;
				case Or:
					code << ".or";
					break;
				case Xor:
					code << ".xor";
					break;
			}
		}
		if constexpr(T::FlushModifier)
		{
			if (this->m_flush)
			{
				code << ".ftz";
			}
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

		code << ", " << m_sourceA->ToString() << ", " << m_sourceB->ToString();
		if (m_sourceC != nullptr)
		{
			code << ", ";
			if (m_negateSourcePredicate)
			{
				code << "!";
			}
			code << m_sourceC->ToString();
		}
		return code.str();
	}

private:
	const Register<PredicateType> *m_destinationP = nullptr;
	const Register<PredicateType> *m_destinationQ = nullptr;
	const Operand<T> *m_sourceA = nullptr;
	const Operand<T> *m_sourceB = nullptr;
	typename T::ComparisonOperator m_comparator;

	const Register<PredicateType> *m_sourceC = nullptr;
	BoolOperator m_boolOperator = And;
	bool m_negateSourcePredicate = false;
};

template<>
class SetPredicateInstruction<Float16Type> : public InstructionStatement, public FlushSubnormalModifier<Float16Type>
{
public:
	enum BoolOperator {
		And,
		Or,
		Xor
	};

	SetPredicateInstruction(const Register<PredicateType> *destination, const Operand<Float16Type> *sourceA, const Operand<Float16Type> *sourceB, Float16Type::ComparisonOperator comparator) : SetPredicateInstruction(destination, sourceA, sourceB, comparator, nullptr, And, false) {}

	SetPredicateInstruction(const Register<PredicateType> *destination, const Operand<Float16Type> *sourceA, const Operand<Float16Type> *sourceB, Float16Type::ComparisonOperator comparator, const Register<PredicateType> *sourceC, BoolOperator boolOperator, bool negateSourcePredicate = false) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_comparator(comparator), m_sourceC(sourceC), m_boolOperator(boolOperator), m_negateSourcePredicate(negateSourcePredicate) {}

	std::string OpCode() const override
	{
		std::ostringstream code;
		code << "setp" << Float16Type::ComparisonOperatorString(m_comparator);
		if (m_sourceC != nullptr)
		{
			switch (m_boolOperator)
			{
				case And:
					code << ".and";
					break;
				case Or:
					code << ".or";
					break;
				case Xor:
					code << ".xor";
					break;
			}
		}
		if (m_flush)
		{
			code << ".ftz";
		}
		code << Float16Type::Name();
		return code.str();
	}

	std::string Operands() const override
	{
		std::ostringstream code;
		code << m_destination->ToString() << ", " << m_sourceA->ToString() << ", " << m_sourceB->ToString();
		if (m_sourceC != nullptr)
		{
			code << ", ";
			if (m_negateSourcePredicate)
			{
				code << "!";
			}
			code << m_sourceC->ToString();
		}
		return code.str();
	}

private:
	const Register<PredicateType> *m_destination = nullptr;
	const Operand<Float16Type> *m_sourceA = nullptr;
	const Operand<Float16Type> *m_sourceB = nullptr;
	Float16Type::ComparisonOperator m_comparator;

	const Register<PredicateType> *m_sourceC = nullptr;
	BoolOperator m_boolOperator = And;
	bool m_negateSourcePredicate = false;
};

}
