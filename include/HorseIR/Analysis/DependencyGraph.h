#pragma once

#include <set>
#include <string>
#include <unordered_map>

#include "HorseIR/Tree/Declaration.h"
#include "HorseIR/Tree/Statements/Statement.h"

namespace HorseIR {

class DependencyGraph
{
public:
	DependencyGraph(const Method *method) : m_method(method) {}

	void InsertDeclaration(const Declaration *declaration);
	void InsertStatement(const Statement *statement);

	void InsertDefinition(const Declaration *declaration, const Statement *statement);
	void InsertDependency(const Statement *statement, const Declaration *declaration);

	std::string ToString() const;

private:
	const Method *m_method = nullptr;

	std::set<const Declaration *> m_declarations;
	std::set<const Statement *> m_statements;

	std::unordered_map<const Declaration *, const Statement *> m_definitions;

	std::unordered_map<const Declaration *, std::set<const Statement *>> m_outgoingEdges;
	std::unordered_map<const Statement *, std::set<const Declaration *>> m_incomingEdges;
};

class GlobalDependencyGraph
{
public:
	void InsertDependencies(const Method *method, DependencyGraph *dependencies);

	std::string ToString() const;

private:
	std::unordered_map<const Method *, DependencyGraph *> m_methodDependencies;
};

}
