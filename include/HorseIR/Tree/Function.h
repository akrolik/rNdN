#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/FunctionDeclaration.h"

#include "HorseIR/Tree/Parameter.h"
#include "HorseIR/Tree/Statements/Statement.h"
#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class SymbolTable;

class Function : public FunctionDeclaration
{
public:
	Function(const std::string& name, const std::vector<Parameter *>& parameters, const std::vector<Type *>& returnTypes, const std::vector<Statement *>& statements, bool kernel = false) : FunctionDeclaration(FunctionDeclaration::Kind::Definition, name), m_parameters(parameters), m_returnTypes(returnTypes), m_statements(statements), m_kernel(kernel) {}

	Function *Clone() const override
	{
		std::vector<Parameter *> parameters;
		for (const auto& parameter : m_parameters)
		{
			parameters.push_back(parameter->Clone());
		}

		std::vector<Type *> returnTypes;
		for (const auto& returnType : m_returnTypes)
		{
			returnTypes.push_back(returnType->Clone());
		}

		std::vector<Statement *> statements;
		for (const auto& statement : m_statements)
		{
			statements.push_back(statement->Clone());
		}

		return new Function(m_name, parameters, returnTypes, statements, m_kernel);
	}

	const std::vector<Parameter *>& GetParameters() const { return m_parameters; }
	Parameter *GetParameter(unsigned int i) const { return m_parameters.at(i); }
	size_t GetParameterCount() const { return m_parameters.size(); }
	void SetParameters(const std::vector<Parameter *>& parameters) { m_parameters = parameters; }

	const std::vector<Type *>& GetReturnTypes() const { return m_returnTypes; }
	Type *GetReturnType(unsigned int i) const { return m_returnTypes.at(i); }
	size_t GetReturnCount() const { return m_returnTypes.size(); }
	void SetReturnType(const std::vector<Type *>& returnTypes) { m_returnTypes = returnTypes; }

	const std::vector<Statement *>& GetStatements() const { return m_statements; }
	void SetStatements(const std::vector<Statement *>& statements) { m_statements = statements; }

	bool IsKernel() const { return m_kernel; }
	void SetKernel(bool kernel) { m_kernel = kernel; }

	SymbolTable *GetSymbolTable() const { return m_symbolTable; }
	void SetSymbolTable(SymbolTable *symbolTable) { m_symbolTable = symbolTable; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			for (auto& parameter : m_parameters)
			{
				parameter->Accept(visitor);
			}
			for (auto& returnType : m_returnTypes)
			{
				returnType->Accept(visitor);
			}
			for (auto& statement : m_statements)
			{
				statement->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			for (const auto& parameter : m_parameters)
			{
				parameter->Accept(visitor);
			}
			for (const auto& returnType : m_returnTypes)
			{
				returnType->Accept(visitor);
			}
			for (const auto& statement : m_statements)
			{
				statement->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

protected:
	std::vector<Parameter *> m_parameters;
	std::vector<Type *> m_returnTypes;
	std::vector<Statement *> m_statements;
	bool m_kernel = false;

	SymbolTable *m_symbolTable = nullptr;
};

}
