#pragma once

#include <stack>
#include <map>

#include "HorseIR/Traversal/ForwardTraversal.h"

namespace HorseIR {

class SymbolTable
{
public:
	friend class SymbolPass_Modules;
	friend class SymbolPass_Imports;
	friend class SymbolPass_Methods;

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

	void Insert(const std::string& name, Entry *symbol);
	Entry *Get(const std::string& name);

	Module *GetModule(const std::string& name);
	MethodDeclaration *GetMethod(const std::string& name);
	Declaration *GetVariable(const std::string& name);

	void AddImport(const std::string& name, Entry *symbol);

	SymbolTable *GetParent() const { return m_parent; }

	std::string ToString() const;

private:
	SymbolTable *m_parent = nullptr;
	std::map<std::string, Entry *> m_table;

	std::map<std::string, Entry *> m_imports;
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

private:
	std::stack<SymbolTable *> m_scopes;
};

class SymbolTableBuilder
{
public:
	void Build(Program *program);
};

class SymbolTableDumper : public ForwardTraversal
{
public:
	using ForwardTraversal::ForwardTraversal;

	void Dump(Program *program);

	void Visit(Program *program) override;
	void Visit(Module *module) override;
	void Visit(Method *method) override;
};

}
