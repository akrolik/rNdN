#pragma once

#include <string>

#include "HorseIR/Tree/Expressions/Expression.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class Identifier : public Expression
{
public:
	Identifier(const std::string& string) : m_string(string) {}

	const Type *GetType() const { return m_type; }
	void SetType(Type *type) { m_type = type; }

	const std::string& GetString() const { return m_string; }

	std::string ToString() const override
	{
		return m_string;
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

private:
	std::string m_string;
	Type *m_type = nullptr;
};

}
