#pragma once

#include "SASS/Operands/Operand.h"

namespace SASS {

class Address : public Operand
{
public:
	Address(const Operand *operand) : m_operand(operand) {}

	std::string ToString() const override
	{
		return "[" + m_operand->ToString() + "]";
	}

private:
	const Operand *m_operand = nullptr;
};

}
