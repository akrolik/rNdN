#pragma once

#include "PTX/Tree/Operands/Operand.h"

namespace PTX {

class StringOperand : public Operand, public DispatchBase
{
public:
	StringOperand(const std::string& string) : m_string(string) {}

	// Properties

	const std::string& GetString() const { return m_string; }
	void SetString(const std::string& string) { m_string = string; }

	// Formatting

	std::string ToString() const override
	{
		return m_string;
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::StringOperand";
		j["string"] = m_string;
		return j;
	}

	// Visitors

	void Accept(OperandVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstOperandVisitor& visitor) const override { visitor.Visit(this); }

private:
	std::string m_string;

};

}
