#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/ModuleContent.h"

#include "HorseIR/Tree/Statements/Statement.h"
#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class Method : public ModuleContent
{
public:
	Method(std::string name, Type *returnType, std::vector<Statement *> statements) : m_name(name), m_returnType(returnType), m_statements(statements) {}

	std::string ToString() const
	{
		std::string code = "def " + m_name + " () : " + m_returnType->ToString() + " {\n";
		for (auto it = m_statements.cbegin(); it != m_statements.cend(); ++it)
		{
			code += "\t\t" + (*it)->ToString() + ";\n";
		}
		return code + "\t}";
	}

private:
	std::string m_name;
	Type *m_returnType;
	std::vector<Statement *> m_statements;
};

}
