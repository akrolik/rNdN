#include "Analysis/Dependency/DependencyAnalysis.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

void DependencyAnalysis::Analyze(const HorseIR::Function *function)
{
	function->Accept(*this);
}

bool DependencyAnalysis::VisitIn(const HorseIR::Function *function)
{
	m_graphOverlay = new FunctionDependencyOverlay(function, m_graph);
	return true;
}

bool DependencyAnalysis::VisitIn(const HorseIR::Statement *statement)
{
	// Add statement to graph, and set as current for identifier dependencies

	m_graph->InsertNode(statement);
	m_graphOverlay->InsertStatement(statement);
	m_currentStatement = statement;
	return true;
}

void DependencyAnalysis::VisitOut(const HorseIR::Statement *statement)
{
	m_currentStatement = nullptr;
}

bool DependencyAnalysis::VisitIn(const HorseIR::AssignStatement *assignS)
{
	// Only traverse the expression, targets do not introduce dependencies
	
	m_graph->InsertNode(assignS);
	m_graphOverlay->InsertStatement(assignS);
	m_currentStatement = assignS;

	assignS->GetExpression()->Accept(*this);
	return false;
}

template<typename T>
void DependencyAnalysis::VisitCompoundStatement(const T *statement)
{
	// Add the statement to the graph

	m_graph->InsertNode(statement);
	m_currentStatement = statement;

	// Organize the contents in an overlay

	m_graphOverlay = new CompoundDependencyOverlay<T>(statement, m_graph, m_graphOverlay);
	m_graphOverlay->InsertStatement(statement);
}

bool DependencyAnalysis::VisitIn(const HorseIR::IfStatement *ifS)
{
	VisitCompoundStatement(ifS);
	return true;
}

bool DependencyAnalysis::VisitIn(const HorseIR::WhileStatement *whileS)
{
	VisitCompoundStatement(whileS);
	return true;
}

bool DependencyAnalysis::VisitIn(const HorseIR::RepeatStatement *repeatS)
{
	VisitCompoundStatement(repeatS);
	return true;
}

void DependencyAnalysis::VisitOut(const HorseIR::IfStatement *ifS)
{
	m_graphOverlay = m_graphOverlay->GetParent();
}

void DependencyAnalysis::VisitOut(const HorseIR::WhileStatement *whileS)
{
	m_graphOverlay = m_graphOverlay->GetParent();
}

void DependencyAnalysis::VisitOut(const HorseIR::RepeatStatement *repeatS)
{
	m_graphOverlay = m_graphOverlay->GetParent();
}

bool DependencyAnalysis::VisitIn(const HorseIR::BlockStatement *blockS)
{
	// Keep block statement organized in the overlay

	m_graphOverlay = new DependencyOverlay(m_graph, m_graphOverlay);
	return true;
}

void DependencyAnalysis::VisitOut(const HorseIR::BlockStatement *blockS)
{
	m_graphOverlay = m_graphOverlay->GetParent();
}

bool DependencyAnalysis::VisitIn(const HorseIR::FunctionLiteral *literal)
{
	// Skip function literal identifier

	return false;
}

bool DependencyAnalysis::VisitIn(const HorseIR::Identifier *identifier)
{
	// Add dependency for each definition

	for (const auto& definition : m_useDefChains.GetDefinitions(identifier))
	{
		m_graph->InsertEdge(definition, m_currentStatement);
	}
	return true;
}

}
