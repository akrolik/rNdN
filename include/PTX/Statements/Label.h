#pragma once

#include "PTX/Operands/Operand.h"
#include "PTX/Statements/Statement.h"

namespace PTX {

class Label : public Statement, public Operand
{
public:
	Label(const std::string& name) : m_name(name) {}

	void SetName(const std::string& name) { m_name = name; }
	std::string GetName() const { return m_name; }

	std::string ToString(unsigned int indentation) const override
	{
		if (indentation > 0)
		{
			return std::string(indentation-1, '\t') + m_name + ":";
		}
		return m_name;
	}

	std::string ToString() const override
	{
		return m_name;
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::Label";
		j["name"] = m_name;
		return j;
	}

	// Visitors

	void Accept(ConstHierarchicalVisitor& visitor) const override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

private:
	std::string m_name;
};

}
