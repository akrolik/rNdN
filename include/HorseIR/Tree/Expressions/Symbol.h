#pragma once

#include <string>

#include "HorseIR/Tree/Expressions/Expression.h"

#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class Symbol : public Expression
{
public:
	Symbol(std::string name) : m_name(name) {}

	std::string ToString() const override
	{
		return m_name;
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

private:
	std::string m_name;
};

}
