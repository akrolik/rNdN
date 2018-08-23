#pragma once

#include <string>

#include "HorseIR/Tree/Expressions/Operand.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Declaration.h"

namespace HorseIR {

class Identifier : public Operand
{
public:
	Identifier(const std::string& string) : Operand(Operand::Kind::Identifier), m_string(string) {}

	const std::string& GetString() const { return m_string; }

	Declaration *GetDeclaration() const { return m_declaration; }
	void SetDeclaration(Declaration *declaration) { m_declaration = declaration; }

	std::string ToString() const override
	{
		return m_string;
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	bool operator==(const Identifier& other) const
	{
		return (m_string == other.m_string && m_declaration == other.m_declaration);
	}

	bool operator!=(const Identifier& other) const
	{
		return !(*this == other);
	}

private:
	std::string m_string;

	Declaration *m_declaration = nullptr;
};

}
