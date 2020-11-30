#pragma once

namespace PTX
{

template<class T, bool R = false, typename Enable = void>
class ComparisonModifier
{
public:
	constexpr static bool Enabled = false;
};

template<class T>
class ComparisonModifier<T, false, std::enable_if_t<is_comparable_type<T>::value>>
{
public:
	constexpr static bool Enabled = true;

	ComparisonModifier(typename T::ComparisonOperator comparisonOperator) : m_comparisonOperator(comparisonOperator) {}

	typename T::ComparisonOperator GetComparisonOperator() const { return m_comparisonOperator; }
	void SetComparisonOperator(typename T::ComparisonOperator comparisonOperator)
	{
	       	m_comparisonOperator = comparisonOperator;
	}

	std::string OpCodeModifier() const
	{
		return T::ComparisonOperatorString(m_comparisonOperator);
	}

protected:
	typename T::ComparisonOperator m_comparisonOperator;
};

}
