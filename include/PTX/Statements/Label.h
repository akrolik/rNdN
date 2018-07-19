#pragma once

#include "PTX/Operands/Operand.h"
#include "PTX/Statements/Statement.h"

namespace PTX {

class Label : public Statement, public Operand
{
public:
	Label(const std::string& name) : m_name(name) {}

	std::string GetName() const { return m_name; }

	std::string ToString(unsigned int indentation = 0) const override
	{
		return std::string(indentation-1, '\t') + m_name;
	}

	std::string ToString() const override
	{
		return m_name;
	}

	std::string Terminator() const override
	{
		return ":";
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::Label";
		j["name"] = m_name;
		return j;
	}

private:
	std::string m_name;
};

}
