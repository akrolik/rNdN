#pragma once

#include <string>

#include "HorseIR/Tree/Expressions/Expression.h"

#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class Identifier : public Expression
{
public:
	Identifier(std::string string) : m_string(string) {}

	std::string GetString() const { return m_string; }

	std::string ToString() const override
	{
		return m_string;
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

private:
	std::string m_string;
};

}
