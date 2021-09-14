#pragma once

#include <string>

#include "PTX/Tree/Declarations/Declaration.h"

#include "PTX/Traversal/Dispatch.h"
#include "PTX/Traversal/ConstFunctionVisitor.h"
#include "PTX/Traversal/FunctionVisitor.h"

namespace PTX {

class Function : public Declaration
{
public:
	Function(const std::string& name, Declaration::LinkDirective linkDirective = Declaration::LinkDirective::None) : Declaration(linkDirective), m_name(name) {};

	// Properties

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	// Formatting

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::Function";
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

	virtual void Accept(FunctionVisitor& visitor) = 0;
	virtual void Accept(ConstFunctionVisitor& visitor) const = 0;

private:
	std::string m_name;
};

}
