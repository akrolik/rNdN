#pragma once

#include "PTX/Tree/Operands/Operand.h"

namespace PTX {

class StringOperand : public Operand
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

private:
	std::string m_string;

};

}
