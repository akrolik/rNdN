#pragma once

#include "SASS/Operands/Operand.h"

#include "Utils/Format.h"

namespace SASS {

class Constant : public Operand
{
public:
	Constant(std::uint32_t bank, std::uint32_t address) : m_bank(bank), m_address(address) {}

	std::string ToString() const override
	{
		return "c[" + Utils::Format::HexString(m_bank) + "][" + Utils::Format::HexString(m_address) + "]";
	}

private:
	std::uint32_t m_bank = 0;
	std::uint32_t m_address = 0;
};

}
