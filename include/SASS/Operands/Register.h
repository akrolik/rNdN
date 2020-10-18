#pragma once

#include <string>

#include "SASS/Operands/Operand.h"

namespace SASS {

class Register : public Operand
{
public:
	Register(const std::string& name) : m_name(name) {}

	std::string ToString() const override
	{
		return m_name;
	}

private:
	std::string m_name;
};

static Register *RZ = new Register("RZ");

}
