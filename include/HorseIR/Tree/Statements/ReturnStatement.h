#pragma once

#include <string>

#include "HorseIR/Tree/Statements/Statement.h"

#include "HorseIR/Tree/Expressions/Identifier.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class ReturnStatement : public Statement
{
public:
	ReturnStatement(Identifier *identifier) : m_identifier(identifier) {}

	Identifier *GetIdentifier() const { return m_identifier; }

	std::string ToString() const override
	{
		return "return " + m_identifier->ToString();
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

private:
	Identifier *m_identifier;
};

}
