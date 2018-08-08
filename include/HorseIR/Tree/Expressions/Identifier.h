#pragma once

#include <string>

#include "HorseIR/Tree/Expressions/Expression.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Declaration.h"

namespace HorseIR {

class Identifier : public Expression
{
public:
	Identifier(const std::string& string) : m_string(string) {}

	const std::string& GetString() const { return m_string; }

	Declaration *GetDeclaration() const { return m_declaration; }
	void SetDeclaration(Declaration *declaration) { m_declaration = declaration; }

	std::string ToString() const override
	{
		return m_string;
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

private:
	std::string m_string;

	Declaration *m_declaration = nullptr;
};

}
