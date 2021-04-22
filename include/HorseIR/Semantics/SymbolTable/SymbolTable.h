#pragma once

#include <string>

#include "Libraries/robin_hood.h"

namespace HorseIR {

class Node;
class Module;
class FunctionDeclaration;
class VariableDeclaration;

class SymbolTable
{
public:
	friend class SymbolPass_Modules;
	friend class SymbolPass_Imports;
	friend class SymbolPass_Functions;
	friend class SymbolTablePrinter;

	struct Symbol
	{
		enum class Kind {
			Module,
			Function,
			Variable
		};

		Symbol(Kind k, const std::string& name, Node *n) : kind(k), name(name), node(n) {}

		friend std::ostream& operator<<(std::ostream& os, const Symbol& value);

		Kind kind;
		std::string name;
		Node *node = nullptr;
	};

	SymbolTable(SymbolTable *parent = nullptr) : m_parent(parent) {}

	SymbolTable *GetImportTable() const { return m_importTable; }
	void SetImportTable(SymbolTable *table) { m_importTable = table; }

	SymbolTable *GetParent() const { return m_parent; }
	void SetParent(SymbolTable *parent) { m_parent = parent; }

	bool ContainsSymbol(const std::string& name) const;
	bool ContainsSymbol(const Symbol *symbol) const;

	Symbol *GetSymbol(const std::string& name, bool assert = true) const;

	void AddSymbol(const std::string& name, Symbol *symbol, bool replace = false);

	const Module *GetModule(const std::string& name) const;
	const FunctionDeclaration *GetFunction(const std::string& name) const;
	const VariableDeclaration *GetVariable(const std::string& name) const;

private:
	SymbolTable *m_parent = nullptr;
	SymbolTable *m_importTable = nullptr;

	robin_hood::unordered_map<std::string, Symbol *> m_table;
};

}
