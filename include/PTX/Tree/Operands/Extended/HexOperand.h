#pragma once

#include "PTX/Tree/Operands/Operand.h"

#include "Utils/Format.h"

namespace PTX {

class HexOperand : public Operand
{
public:
	HexOperand(unsigned int value) : m_value(value) {}

	// Properties

	unsigned int GetValue() const { return m_value; }
	void SetValue(unsigned int value) { m_value = value; }

	// Formatting

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

	void Accept(OperandVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstOperandVisitor& visitor) const override { visitor.Visit(this); }

private:
	unsigned int m_value;

};

}
