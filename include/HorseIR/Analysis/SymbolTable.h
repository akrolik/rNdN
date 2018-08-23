#pragma once

#include <stack>
#include <unordered_map>

#include "HorseIR/Traversal/ConstForwardTraversal.h"
#include "HorseIR/Traversal/ForwardTraversal.h"

namespace HorseIR {

class SymbolTable
{
public:
	friend class SymbolPass_Modules;
	friend class SymbolPass_Imports;
	friend class SymbolPass_Methods;
	friend class SymbolTableDumper;

	struct Entry
	{
		enum class Kind {
			Module,
			Method,
			Variable
		};

		Entry(Kind k, Node *n) : kind(k), node(n) {}

		std::string ToString() const;

		Kind kind;
		Node *node = nullptr;
	};

	SymbolTable(SymbolTable *parent = nullptr) : m_parent(parent) {}

	SymbolTable *GetParent() const { return m_parent; }

	Entry *GetSymbol(const std::string& name);

	Module *GetModule(const std::string& name);
	MethodDeclaration *GetMethod(const std::string& name);
	Declaration *GetVariable(const std::string& name);

	void AddSymbol(const std::string& name, Entry *symbol);
	void AddImport(const std::string& name, Entry *symbol);

private:
	SymbolTable *m_parent = nullptr;

	std::unordered_map<std::string, Entry *> m_table;
	std::unordered_map<std::string, Entry *> m_imports;
};

class SymbolPass_Modules : public ForwardTraversal
{
public:
	using ForwardTraversal::ForwardTraversal;

	void Build(Program *program);

	void Visit(Module *module) override;
	void Visit(BuiltinMethod *method) override;
	void Visit(Method *method) override;

private:
	std::stack<SymbolTable *> m_scopes;
};

class SymbolPass_Imports : public ForwardTraversal
{
public:
	using ForwardTraversal::ForwardTraversal;

	void Build(Program *program);
	
	void Visit(Module *module) override;
	void Visit(Import *import) override;
	void Visit(Method *method) override; // Stops the traversal

private:
	Module *m_module = nullptr;
};

class SymbolPass_Methods : public ForwardTraversal
{
public:
	using ForwardTraversal::ForwardTraversal;

	void Build(Program *program);

	void Visit(Program *program) override;
	void Visit(Module *module) override;
	void Visit(Method *method) override;

	void Visit(Declaration *declaration) override;
	void Visit(CallExpression *call) override;
	void Visit(Identifier *identifier) override;

	void Visit(FunctionLiteral *literal) override;

private:
	MethodDeclaration *LookupIdentifier(const ModuleIdentifier *identifier);

	std::stack<SymbolTable *> m_scopes;
};

class SymbolTableBuilder
{
public:
	void Build(Program *program);
};

class SymbolTableDumper : public ConstForwardTraversal
{
public:
	using ConstForwardTraversal::ConstForwardTraversal;

	void Dump(const Program *program);

	void Visit(const Program *program) override;
	void Visit(const Module *module) override;
	void Visit(const Method *method) override;

	void Visit(const Declaration *declaration) override;

private:
	std::stack<SymbolTable *> m_scopes;
};

}
