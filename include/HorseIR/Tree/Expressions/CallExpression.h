#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/Expressions/Expression.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/MethodDeclaration.h"
#include "HorseIR/Tree/Expressions/ModuleIdentifier.h"
#include "HorseIR/Tree/Expressions/Operand.h"

namespace HorseIR {

class CallExpression : public Expression
{
public:
	CallExpression(ModuleIdentifier *identifier, const std::vector<Operand *>& arguments) : m_identifier(identifier), m_arguments(arguments) {}

	ModuleIdentifier *GetIdentifier() const { return m_identifier; }
	void SetIdentifier(ModuleIdentifier *identifier) { m_identifier = identifier; }

	const std::vector<Operand *>& GetArguments() const { return m_arguments; }
	Operand *GetArgument(unsigned int index) const { return m_arguments.at(index); }

	MethodDeclaration *GetMethod() const { return m_method; }
	void SetMethod(MethodDeclaration *method) { m_method = method; }

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
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

private:
	ModuleIdentifier *m_identifier = nullptr;
	std::vector<Operand *> m_arguments;

	MethodDeclaration *m_method = nullptr;
};

}
