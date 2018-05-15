#pragma once

#include <sstream>

#include "PTX/Statements/InstructionStatement.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class SetPredicateInstruction : public InstructionStatement
{
	static_assert(std::is_base_of<ValueType, T>::value, "T must be a PTX::ValueType");
public:
	enum BoolOperator {
		And,
		Or,
		Xor
	};

	enum ComparisonOperator {
		Equal,
		NotEqual,
		LessThan,
		GreaterThan,
		LessEqual,
		GreaterEqual
	};

	SetPredicateInstruction(Register<PredicateType> *destination, Operand<T> *sourceA, Operand<T> *sourceB, ComparisonOperator comparator) : SetPredicateInstruction(destination, nullptr, sourceA, sourceB, comparator, nullptr, And, false) {}

	SetPredicateInstruction(Register<PredicateType> *destinationP, Register<PredicateType> *destinationQ, Operand<T> *sourceA, Operand<T> *sourceB, ComparisonOperator comparator, Register<PredicateType> *sourceC, BoolOperator boolOperator, bool negateSourcePredicate = nullptr) : m_destinationP(destinationP), m_destinationQ(destinationQ), m_sourceA(sourceA), m_sourceB(sourceB), m_comparator(comparator), m_sourceC(sourceC), m_boolOperator(boolOperator), m_negateSourcePredicate(negateSourcePredicate) {}

	std::string OpCode() const
	{
		std::ostringstream code;
		code << "setp";
		switch (m_comparator)
		{
			case Equal:
				code << ".eq";
				break;
			case NotEqual:
				code << ".ne";
				break;
			case LessThan:
				code << ".lt";
				break;
			case GreaterThan:
				code << ".gt";
				break;
			case LessEqual:
				code << ".le";
				break;
			case GreaterEqual:
				code << ".ge";
				break;
		}
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
	ComparisonOperator m_comparator;

	Register<PredicateType> *m_sourceC = nullptr;
	BoolOperator m_boolOperator = And;
	bool m_negateSourcePredicate = false;
};

}
