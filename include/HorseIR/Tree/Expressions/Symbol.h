#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/Expressions/Expression.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Types/BasicType.h"

namespace HorseIR {

class Symbol : public Expression
{
public:
	Symbol(const std::string& name, BasicType *type) : m_names({name}), m_literalType(type) {}
	Symbol(const std::vector<std::string>& names, BasicType *type) : m_names(names), m_literalType(type) {}

	const BasicType *GetLiteralType() const { return m_literalType; }

	const std::vector<std::string>& GetNames() const { return m_names; }
	const std::string& GetName(unsigned int index) const { return m_names.at(index); }

	unsigned int GetCount() const { return m_names.size(); }

	std::string ToString() const override
	{
		std::string code;
		if (m_names.size() > 1)
		{
			code += "(";
		}
		bool first = true;
		for (const auto& name : m_names)
		{
			if (!first)
			{
				code += ", ";
			}
			first = false;
			code += "`" + name;
		}
		if (m_names.size() > 1)
		{
			code += ")";
		}
		return code + ":" + m_literalType->ToString();
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

private:
	std::vector<std::string> m_names;
	BasicType *m_literalType = nullptr;
};

}
