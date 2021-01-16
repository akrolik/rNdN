#pragma once

#include "PTX/Tree/Operands/Operand.h"

namespace PTX {

class Label : public Operand
{
public:
	Label(const std::string& name) : m_name(name) {}

	// Properties

	void SetName(const std::string& name) { m_name = name; }
	const std::string& GetName() const { return m_name; }

	// Formatting

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

	using Operand::Accept;

	void Accept(OperandVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstOperandVisitor& visitor) const override { visitor.Visit(this); }

private:
	std::string m_name;
};

}
