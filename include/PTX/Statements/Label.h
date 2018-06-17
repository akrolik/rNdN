#pragma once

#include "PTX/Statements/Statement.h"

namespace PTX {

class Label : public Statement
{
public:
	Label(const std::string& name) : m_name(name) {}

	std::string GetName() const { return m_name; }

	std::string ToString() const override
	{
		return m_name;
	}

	std::string Terminator() const override
	{
		return ":";
	}
private:
	std::string m_name;
};

}
