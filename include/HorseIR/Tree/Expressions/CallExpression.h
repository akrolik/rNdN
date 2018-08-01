#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/Expressions/Expression.h"

#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/Expressions/ModuleIdentifier.h"
#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class CallExpression : public Expression
{
public:
	CallExpression(ModuleIdentifier *identifier, const std::vector<Expression *>& arguments) : m_identifier(identifier), m_arguments(arguments) {}

	const Type *GetType() const { return nullptr; }

	ModuleIdentifier *GetIdentifier() const { return m_identifier; }
	void SetIdentifier(ModuleIdentifier *identifier) { m_identifier = identifier; }

	const std::vector<Expression *>& GetArguments() const { return m_arguments; }
	Expression *GetArgument(unsigned int index) const { return m_arguments.at(index); }

	Method *GetMethod() const { return m_method; }
	void SetMethod(Method *method) { m_method = method; }

	std::string ToString() const override
	{
		std::string code = "@" + m_identifier->ToString() + "(";
		bool first = true;
		for (const auto& argument : m_arguments)
		{
			if (!first)
			{
				code += ", ";
			}
			first = false;
			code += argument->ToString();
		}
		return code + ")";
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

private:
	ModuleIdentifier *m_identifier = nullptr;
	std::vector<Expression *> m_arguments;

	Method *m_method = nullptr;
};

}
