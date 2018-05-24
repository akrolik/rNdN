#pragma once

#include <sstream>

#include "PTX/Statements/InstructionStatement.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class SetPredicateInstruction : public InstructionStatement
{
	REQUIRE_BASE_TYPE(SetPredicateInstruction, ScalarType);
	DISABLE_EXACT_TYPE(SetPredicateInstruction, Bit8Type);
	DISABLE_EXACT_TYPE(SetPredicateInstruction, Int8Type);
	DISABLE_EXACT_TYPE(SetPredicateInstruction, UInt8Type);
public:
	enum BoolOperator {
		And,
		Or,
		Xor
	};

	SetPredicateInstruction(Register<PredicateType> *destination, Operand<T> *sourceA, Operand<T> *sourceB, typename T::ComparisonOperator comparator) : SetPredicateInstruction(destination, nullptr, sourceA, sourceB, comparator, nullptr, And, false) {}

	SetPredicateInstruction(Register<PredicateType> *destinationP, Register<PredicateType> *destinationQ, Operand<T> *sourceA, Operand<T> *sourceB, typename T::ComparisonOperator comparator, Register<PredicateType> *sourceC, BoolOperator boolOperator, bool negateSourcePredicate = false) : m_destinationP(destinationP), m_destinationQ(destinationQ), m_sourceA(sourceA), m_sourceB(sourceB), m_comparator(comparator), m_sourceC(sourceC), m_boolOperator(boolOperator), m_negateSourcePredicate(negateSourcePredicate) {}

	std::string OpCode() const
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
		code << T::Name();
		return code.str();
	}

	std::string Operands() const
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
	Register<PredicateType> *m_destinationP = nullptr;
	Register<PredicateType> *m_destinationQ = nullptr;
	Operand<T> *m_sourceA = nullptr;
	Operand<T> *m_sourceB = nullptr;
	typename T::ComparisonOperator m_comparator;

	Register<PredicateType> *m_sourceC = nullptr;
	BoolOperator m_boolOperator = And;
	bool m_negateSourcePredicate = false;
};

template<Bits B, unsigned int N>
class SetPredicateInstruction<FloatType<B, N>> : public InstructionStatement, public FlushSubnormalModifier
{
public:
	enum BoolOperator {
		And,
		Or,
		Xor
	};

	SetPredicateInstruction(Register<PredicateType> *destination, Operand<FloatType<B, N>> *sourceA, Operand<FloatType<B, N>> *sourceB, typename FloatType<B, N>::ComparisonOperator comparator) : SetPredicateInstruction(destination, nullptr, sourceA, sourceB, comparator, nullptr, And, false) {}

	SetPredicateInstruction(Register<PredicateType> *destinationP, Register<PredicateType> *destinationQ, Operand<FloatType<B, N>> *sourceA, Operand<FloatType<B, N>> *sourceB, typename FloatType<B, N>::ComparisonOperator comparator, Register<PredicateType> *sourceC, BoolOperator boolOperator, bool negateSourcePredicate = false) : m_destinationP(destinationP), m_destinationQ(destinationQ), m_sourceA(sourceA), m_sourceB(sourceB), m_comparator(comparator), m_sourceC(sourceC), m_boolOperator(boolOperator), m_negateSourcePredicate(negateSourcePredicate) {}

	std::string OpCode() const
	{
		std::ostringstream code;
		code << "setp" << FloatType<B, N>::ComparisonOperatorString(m_comparator);
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
		code << FloatType<B, N>::Name();
		return code.str();
	}

	std::string Operands() const
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
	Register<PredicateType> *m_destinationP = nullptr;
	Register<PredicateType> *m_destinationQ = nullptr;
	Operand<FloatType<B, N>> *m_sourceA = nullptr;
	Operand<FloatType<B, N>> *m_sourceB = nullptr;
	typename FloatType<B, N>::ComparisonOperator m_comparator;

	Register<PredicateType> *m_sourceC = nullptr;
	BoolOperator m_boolOperator = And;
	bool m_negateSourcePredicate = false;
};

template<>
class SetPredicateInstruction<Float16Type> : public InstructionStatement, public FlushSubnormalModifier
{
public:
	enum BoolOperator {
		And,
		Or,
		Xor
	};

	SetPredicateInstruction(Register<PredicateType> *destination, Operand<Float16Type> *sourceA, Operand<Float16Type> *sourceB, Float16Type::ComparisonOperator comparator) : SetPredicateInstruction(destination, sourceA, sourceB, comparator, nullptr, And, false) {}

	SetPredicateInstruction(Register<PredicateType> *destination, Operand<Float16Type> *sourceA, Operand<Float16Type> *sourceB, Float16Type::ComparisonOperator comparator, Register<PredicateType> *sourceC, BoolOperator boolOperator, bool negateSourcePredicate = false) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_comparator(comparator), m_sourceC(sourceC), m_boolOperator(boolOperator), m_negateSourcePredicate(negateSourcePredicate) {}

	std::string OpCode() const
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

	std::string Operands() const
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
	Register<PredicateType> *m_destination = nullptr;
	Operand<Float16Type> *m_sourceA = nullptr;
	Operand<Float16Type> *m_sourceB = nullptr;
	Float16Type::ComparisonOperator m_comparator;

	Register<PredicateType> *m_sourceC = nullptr;
	BoolOperator m_boolOperator = And;
	bool m_negateSourcePredicate = false;
};

}
