#pragma once

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
	PredicateModifier(const Register<PredicateType> *sourceC, BoolOperator boolOperator, bool negateSourcePredicte = false) : m_boolOperator(boolOperator) {}

	std::string OpCodeModifier() const
	{
		if (m_sourceC != nullptr)
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

	std::string OperandsModifier() const
	{
		if (m_sourceC != nullptr)
		{
			if (m_negateSourcePredicate)
			{
				return ", !" + m_sourceC->ToString();
			}
			return ", " + m_sourceC->ToString();
		}
		return "";
	}

protected:
	const Register<PredicateType> *m_sourceC = nullptr;
	BoolOperator m_boolOperator = BoolOperator::And;
	bool m_negateSourcePredicate = false;
};

}
