#pragma once

#include "PTX/Tree/Operands/Operand.h"

#include "Utils/Format.h"

namespace PTX {

class HexOperand : public Operand
{
public:
	HexOperand(unsigned int value) : m_value(value) {}

	std::string ToString() const override
	{
		return Utils::Format::HexString(m_value);
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
