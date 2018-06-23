#pragma once

#include "PTX/Operands/Operand.h"

namespace PTX {

class StringOperand : public Operand
{
public:
	StringOperand(const std::string& string) : m_string(string) {}

	std::string ToString() const override
	{
		return m_string;
	}

private:
	std::string m_string;

};

}
