#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/ModuleContent.h"

#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Parameter.h"
#include "HorseIR/Tree/Statements/Statement.h"
#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class SymbolTable;

class Method : public ModuleContent
{
public:
	Method(const std::string& name, const std::vector<Parameter *>& parameters, Type *returnType, const std::vector<Statement *>& statements, bool kernel = false) : m_name(name), m_parameters(parameters), m_returnType(returnType), m_statements(statements), m_kernel(kernel) {}

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	const std::vector<Parameter *>& GetParameters() const { return m_parameters; }
	Type* GetReturnType() const { return m_returnType; }
	const std::vector<Statement *>& GetStatements() const { return m_statements; }

	bool IsKernel() const { return m_kernel; }
	void SetKernel(bool kernel) { m_kernel = kernel; }

	SymbolTable *GetSymbolTable() const { return m_symbolTable; }
	void SetSymbolTable(SymbolTable *symbolTable) { m_symbolTable = symbolTable; }

	std::string ToString() const override
	{
		std::string code = std::string((m_kernel) ? "kernel" : "def") + " " + m_name + " (";
		bool first = true;
		for (const auto& parameter : m_parameters)
		{
			if (!first)
			{
				code += ", ";
			}
			first = false;
			code += parameter->ToString();
		}
		code += ") : " + m_returnType->ToString() + " {\n";
		for (const auto& statement : m_statements)
		{
			code += "\t\t" + statement->ToString() + ";\n";
		}
		return code + "\t}";
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

private:
	std::string m_name;
	std::vector<Parameter *> m_parameters;
	Type *m_returnType;
	std::vector<Statement *> m_statements;
	bool m_kernel = false;

	SymbolTable *m_symbolTable = nullptr;
};

}
