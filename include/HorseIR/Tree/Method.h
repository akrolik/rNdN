#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/ModuleContent.h"

#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Statements/Statement.h"
#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class Method : public ModuleContent
{
public:
	Method(std::string name, Type *returnType, std::vector<Statement *> statements) : m_name(name), m_returnType(returnType), m_statements(statements) {}

	Type* GetReturnType() const { return m_returnType; }
	const std::vector<Statement *>& GetStatements() const { return m_statements; }

	std::string ToString() const override
	{
		std::string code = "def " + m_name + " () : " + m_returnType->ToString() + " {\n";
		for (auto statement : m_statements)
		{
			code += "\t\t" + statement->ToString() + ";\n";
		}
		return code + "\t}";
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

private:
	std::string m_name;
	Type *m_returnType;
	std::vector<Statement *> m_statements;
};

}
