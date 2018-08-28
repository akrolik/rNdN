#include "HorseIR/Analysis/DependencyAnalysis.h"

#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/Declaration.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"
#include "HorseIR/Tree/Statements/ReturnStatement.h"
#include "HorseIR/Tree/Expressions/Identifier.h"

namespace HorseIR {

void DependencyAnalysis::Analyze(const Program *program)
{
	program->Accept(*this);
}

void DependencyAnalysis::Visit(const Method *method)
{
	m_dependencies = new DependencyGraph(method);
	m_graph->InsertDependencies(method, m_dependencies);
	ConstForwardTraversal::Visit(method);
	m_dependencies = nullptr;
}

void DependencyAnalysis::Visit(const Declaration *declaration)
{
	m_dependencies->InsertDeclaration(declaration);
}

void DependencyAnalysis::Visit(const AssignStatement *assign)
{
	auto variable = assign->GetDeclaration();
	variable->Accept(*this);
	m_dependencies->InsertStatement(assign);
	m_dependencies->InsertDefinition(variable, assign);

	m_currentStatement = assign;
	ConstForwardTraversal::Visit(assign);
	m_currentStatement = nullptr;
}

void DependencyAnalysis::Visit(const ReturnStatement *ret)
{
	m_dependencies->InsertStatement(ret);

	m_currentStatement = ret;
	ConstForwardTraversal::Visit(ret);
	m_currentStatement = nullptr;
}

void DependencyAnalysis::Visit(const Identifier *identifier)
{
	m_dependencies->InsertDependency(m_currentStatement, identifier->GetDeclaration());
}

}
