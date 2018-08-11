#pragma once

#include "HorseIR/Tree/Expressions/Literals/Literal.h"

#include <string>
#include <vector>

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Types/BasicType.h"

namespace HorseIR {

class SymbolLiteral : public Literal<std::string>
{
public:
	SymbolLiteral(const std::string& value) : Literal<std::string>(value, new BasicType(BasicType::Kind::Symbol)) {}
	SymbolLiteral(const std::vector<std::string>& values) : Literal<std::string>(values, new BasicType(BasicType::Kind::Symbol)) {}

	std::string ToString() const override
	{
		std::string code;
		if (m_values.size() > 1)
		{
			code += "(";
		}
		bool first = true;
		for (const auto& value : m_values)
		{
			if (!first)
			{
				code += ", ";
			}
			code += "`" + value;
		}
		if (m_values.size() > 1)
		{
			code += ")";
		}
		return code + ":" + m_type->ToString();
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }
};

}
