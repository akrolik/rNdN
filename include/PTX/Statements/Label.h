#pragma once

#include "PTX/Statements/Statement.h"

namespace PTX {

class Label : public Statement
{
public:
	Label(std::string name) : m_name(name) {}

	std::string GetName() const
	{
		return m_name;
	}

	std::string ToString() const
	{
		return m_name;
	}

	std::string Terminator() const
	{
		return ":";
	}
private:
	std::string m_name;
};

}
