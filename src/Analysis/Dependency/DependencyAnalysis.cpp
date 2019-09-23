#include "Analysis/Dependency/DependencyAnalysis.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

void DependencyAnalysis::Build(const HorseIR::Function *function)
{
	function->Accept(*this);
}

bool DependencyAnalysis::VisitIn(const HorseIR::Function *function)
{
	auto functionOverlay = new FunctionDependencyOverlay(function, m_graph);
	m_graphOverlay = new DependencyOverlay(m_graph, functionOverlay);
	return true;
}

void DependencyAnalysis::VisitOut(const HorseIR::Function *function)
{
	m_graphOverlay = m_graphOverlay->GetParent();
}

bool DependencyAnalysis::VisitIn(const HorseIR::Statement *statement)
{
	m_currentStatement = statement;
	m_graph->InsertNode(statement);
	m_graphOverlay->InsertStatement(statement);
	return true;
}

void DependencyAnalysis::VisitOut(const HorseIR::Statement *statement)
{
	m_currentStatement = nullptr;
}

template<typename T>
void DependencyAnalysis::VisitCompoundStatement(const typename T::NodeType *statement)
{
	// Add the statement to the dependency graph

	m_graph->InsertNode(statement);
	m_currentStatement = statement;

	// Organize its contents in an overlay

	m_graphOverlay = new T(statement, m_graph, m_graphOverlay);
	m_graphOverlay->InsertStatement(statement);
}

bool DependencyAnalysis::VisitIn(const HorseIR::IfStatement *ifS)
{
	VisitCompoundStatement<IfDependencyOverlay>(ifS);
	return true;
}

bool DependencyAnalysis::VisitIn(const HorseIR::WhileStatement *whileS)
{
	VisitCompoundStatement<WhileDependencyOverlay>(whileS);
	return true;
}

bool DependencyAnalysis::VisitIn(const HorseIR::RepeatStatement *repeatS)
{
	VisitCompoundStatement<RepeatDependencyOverlay>(repeatS);
	return true;
}

void DependencyAnalysis::VisitOut(const HorseIR::IfStatement *ifS)
{
	m_graphOverlay = m_graphOverlay->GetParent();
	m_currentStatement = nullptr;
}

void DependencyAnalysis::VisitOut(const HorseIR::WhileStatement *whileS)
{
	m_graphOverlay = m_graphOverlay->GetParent();
	m_currentStatement = nullptr;
}

void DependencyAnalysis::VisitOut(const HorseIR::RepeatStatement *repeatS)
{
	m_graphOverlay = m_graphOverlay->GetParent();
	m_currentStatement = nullptr;
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
 
bool DependencyAnalysis::VisitIn(const HorseIR::AssignStatement *assignS)
{
	// Traverse the LHS and RHS of the assignment, keeping track if the identifiers
	// are either reads or writes (targets)

	m_currentStatement = assignS;
	m_graph->InsertNode(assignS);
	m_graphOverlay->InsertStatement(assignS);
	
	m_isTarget = true;
	for (const auto& target : assignS->GetTargets())
	{
		// Only traverse the identifiers, declarations do not introduce dependencies

		target->Accept(*this);
	}
	m_isTarget = false;
	assignS->GetExpression()->Accept(*this);

	m_currentStatement = nullptr;
	return false;
}

bool DependencyAnalysis::VisitIn(const HorseIR::FunctionLiteral *literal)
{
	// Skip identifier

	return false;
}

bool DependencyAnalysis::VisitIn(const HorseIR::Identifier *identifier)
{
	// Get the incoming reads and writes (for which we can have dependencies)

	auto symbol = identifier->GetSymbol();
	const auto& [reads, writes] = m_accessAnalysis.GetInSet(m_currentStatement);

	// Add the flow, anti, and output dependency edges to the graph

	if (m_isTarget)
	{
		// Get anti-dependencies with previous reads

		if (reads.find(symbol) != reads.end())
		{
			for (const auto& read : *reads.at(symbol))
			{
				bool isBackEdge = (m_graph->ContainsNode(read) == false || read == m_currentStatement);
				m_graph->InsertEdge(read, m_currentStatement, symbol, isBackEdge);
			}
		}
	}

	// Get flow/output dependencies with previous writes

	if (writes.find(symbol) != writes.end())
	{
		for (const auto& write : *writes.at(symbol))
		{
			bool isBackEdge = (m_graph->ContainsNode(write) == false || write == m_currentStatement);
			m_graph->InsertEdge(write, m_currentStatement, symbol, isBackEdge);
		}
	}

	return true;
}

}
