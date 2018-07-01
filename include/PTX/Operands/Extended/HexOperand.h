#pragma once

#include "PTX/Operands/Operand.h"

namespace PTX {

class HexOperand : public Operand
{
public:
	HexOperand(unsigned int value) : m_value(value) {}

	std::string ToString() const override
	{
		std::ostringstream hex;
		hex << std::hex << m_value;
		return std::string("0x") + hex.str();
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::HexOperand";
		j["value"] = ToString();
		return j;
	}

private:
	unsigned int m_value;

};

}
