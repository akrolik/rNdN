#pragma once

#include "HorseIR/Tree/Expressions/Expression.h"

namespace HorseIR {

class Symbol : public Expression
{
public:
	Symbol(std::string name) : m_name(name) {}

	std::string ToString() const
	{
		return m_name;
	}

private:
	std::string m_name;
};

}
