#include "Analysis/Dependency/DependencyAnalysis.h"

#include "Analysis/Dependency/Overlay/DependencyOverlayPrinter.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

void DependencyAnalysis::Build(const HorseIR::Function *function)
{
	auto timeDependencies_start = Utils::Chrono::Start();
	function->Accept(*this);
	auto timeDependencies = Utils::Chrono::End(timeDependencies_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_outline_graph))
	{
		Utils::Logger::LogInfo("Dependency graph analysis");

		auto dependencyString = Analysis::DependencyOverlayPrinter::PrettyString(m_graphOverlay);
		Utils::Logger::LogInfo(dependencyString, 0, true, Utils::Logger::NoPrefix);
	}

	Utils::Logger::LogTiming("Dependency graph analysis", timeDependencies);
}

bool DependencyAnalysis::VisitIn(const HorseIR::Function *function)
{
	m_functionOverlay = new FunctionDependencyOverlay(function, m_graph);
	m_graphOverlay = new DependencyOverlay(m_graph, m_functionOverlay);
	return true;
}

void DependencyAnalysis::VisitOut(const HorseIR::Function *function)
{
	m_graphOverlay = m_graphOverlay->GetParent();
}

bool DependencyAnalysis::VisitIn(const HorseIR::Statement *statement)
{
	m_currentStatement = statement;

	auto isGPU = m_gpuHelper.IsGPU(statement);
	m_graph->InsertNode(statement, isGPU);
	m_graphOverlay->InsertStatement(statement);

	// Reset operand counter

	m_index = 0;
	return true;
}

void DependencyAnalysis::VisitOut(const HorseIR::Statement *statement)
{
	m_currentStatement = nullptr;
}

template<typename T>
void DependencyAnalysis::VisitCompoundStatement(const typename T::NodeType *statement)
{
	m_currentStatement = statement;

	// Add the statement to the dependency graph

	m_graph->InsertNode(statement, false);

	// Organize its contents in an overlay

	m_graphOverlay = new T(statement, m_graph, m_graphOverlay);
	m_graphOverlay->InsertStatement(statement);

	// Reset operand counter

	m_index = 0;
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

	auto isGPU = m_gpuHelper.IsGPU(assignS);
	m_graph->InsertNode(assignS, isGPU);
	m_graphOverlay->InsertStatement(assignS);
	
	m_isTarget = true;
	m_index = 0;
	for (const auto& target : assignS->GetTargets())
	{
		// Only traverse the identifiers, declarations do not introduce dependencies

		target->Accept(*this);
	}

	m_isTarget = false;
	m_index = 0;
	assignS->GetExpression()->Accept(*this);

	m_currentStatement = nullptr;
	return false;
}

bool DependencyAnalysis::VisitIn(const HorseIR::FunctionLiteral *literal)
{
	// Skip identifier

	return false;
}

void DependencyAnalysis::VisitOut(const HorseIR::FunctionLiteral *literal)
{
	// Do nothing
}

void DependencyAnalysis::VisitOut(const HorseIR::Expression *expression)
{
	// For each expression, we increment the index. This allows us to track where in the function call we are

	m_index++;
}

bool DependencyAnalysis::VisitIn(const HorseIR::Identifier *identifier)
{
	// Get the incoming reads and writes (for which we can have dependencies)

	auto symbol = identifier->GetSymbol();
	const auto& [reads, writes] = m_accessAnalysis.GetInSet(m_currentStatement);

	// Add the flow, anti, and output dependency edges to the graph

	if (m_isTarget)
	{
		// Get anti-dependencies with previous reads, never synchronized

		if (reads.find(symbol) != reads.end())
		{
			for (const auto& read : *reads.at(symbol))
			{
				bool isBackEdge = (m_graph->ContainsNode(read) == false || read == m_currentStatement);
				m_graph->InsertEdge(read, m_currentStatement, symbol, isBackEdge, false);
			}
		}
	}

	// Get flow/output dependencies with previous writes

	if (writes.find(symbol) != writes.end())
	{
		for (const auto& write : *writes.at(symbol))
		{
			// Synchronization occurs between the two statements at the current operand index

			bool isBackEdge = (m_graph->ContainsNode(write) == false || write == m_currentStatement);
			bool isSynchronized = m_gpuHelper.IsSynchronized(write, m_currentStatement, m_index);

			m_graph->InsertEdge(write, m_currentStatement, symbol, isBackEdge, isSynchronized);
		}
	}

	return true;
}

}
