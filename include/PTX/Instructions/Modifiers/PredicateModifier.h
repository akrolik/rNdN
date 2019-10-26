#pragma once

#include "PTX/Operands/Extended/InvertedOperand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

class PredicateModifier
{
public:
	enum class BoolOperator {
		And,
		Or,
		Xor
	};

	PredicateModifier() {}
	PredicateModifier(const Register<PredicateType> *sourcePredicate, BoolOperator boolOperator, bool negateSourcePredicate = false) : m_sourcePredicate(sourcePredicate), m_boolOperator(boolOperator), m_negateSourcePredicate(negateSourcePredicate) {}

	const Register<PredicateType> *GetSourcePredicate() const { return m_sourcePredicate; }
	void SetSourcePredicate(const Register<PredicateType> *source) { m_sourcePredicate; }

	BoolOperator GetBoolOperator() const { return m_boolOperator; }
	void SetBoolOperator(BoolOperator boolOperator) { m_boolOperator = boolOperator; }

	bool GetNegateSourcePredicate() const { return m_negateSourcePredicate; }
	void SetNegateSourcePredicate(bool negateSourcePredicate) { m_negateSourcePredicate = negateSourcePredicate; }

	std::string OpCodeModifier() const
	{
		if (m_sourcePredicate != nullptr)
		{
			switch (m_boolOperator)
			{
				case BoolOperator::And:
					return ".and";
				case BoolOperator::Or:
					return ".or";
				case BoolOperator::Xor:
					return ".xor";
			}
			return ".<unknown>";
		}
		return "";
	}

	const Operand *OperandsModifier() const
	{
		if (m_sourcePredicate != nullptr)
		{
			if (m_negateSourcePredicate)
			{
				return new InvertedOperand(m_sourcePredicate);
			}
			return m_sourcePredicate;
		}
		return nullptr;
	}

protected:
	const Register<PredicateType> *m_sourcePredicate = nullptr;
	BoolOperator m_boolOperator = BoolOperator::And;
	bool m_negateSourcePredicate = false;
};

}
