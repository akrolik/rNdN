#pragma once

#include "SASS/Operands/Operand.h"

#include "Utils/Format.h"

namespace SASS {

class Immediate : public Operand
{
public:
	Immediate(std::uint32_t value) : m_value(value) {}

	std::string ToString() const override
	{
		return Utils::Format::HexString(m_value);
	}

private:
	std::uint32_t m_value = 0;
};

}
