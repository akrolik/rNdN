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

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor& visitor) override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor& visitor) const override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

private:
	std::string m_name;
};

}
