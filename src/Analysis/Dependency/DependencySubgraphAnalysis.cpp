#include "Analysis/Dependency/DependencySubgraphAnalysis.h"

#include "Analysis/Dependency/Overlay/DependencyOverlay.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

void DependencySubgraphAnalysis::Analyze(DependencyOverlay *overlay)
{
	overlay->Accept(*this);
}

const DependencyOverlay *DependencySubgraphAnalysis::GetScopedOverlay(const DependencyOverlay *containerOverlay, const HorseIR::Statement *statement) const
{
	// Check if the current container directly contains the statement

	if (containerOverlay->ContainsStatement(statement))
	{
		return containerOverlay;
	}

	// Recursively check if the statement is found in any child overlay

	for (const auto& overlay : containerOverlay->GetChildren())
	{
		if (GetScopedOverlay(overlay, statement) != nullptr)
		{
			return overlay;
		}
	}

	return nullptr;
}

void DependencySubgraphAnalysis::InsertEdge(DependencySubgraph *subgraph, const DependencySubgraphNode& source, const DependencySubgraphNode& destination, bool isBackEdge, const std::unordered_set<const HorseIR::SymbolTable::Symbol *>& symbols)
{
	subgraph->InsertEdge(source, destination, symbols, isBackEdge);
}

void DependencySubgraphAnalysis::ProcessOverlay(const DependencyOverlay *overlay, const DependencyOverlay *containerOverlay)
{
	// Convenience variables for setting up the graphs

	auto graph = containerOverlay->GetGraph();
	auto subgraph = containerOverlay->GetSubgraph();

	// Check for each statement in the overlay, all corresponding edges for crossing boundaries

	for (const auto& statement : overlay->GetStatements())
	{
		auto statementOverlay = GetScopedOverlay(containerOverlay, statement);

		for (const auto& successor : graph->GetSuccessors(statement))
		{
			// Get the overlay for the successor and check that it is within the container

			auto successorOverlay = GetScopedOverlay(containerOverlay, successor);
			if (successorOverlay == nullptr)
			{
				continue;
			}


			// Add edge to the subgraph with the symbol data

			auto isBackEdge = graph->IsBackEdge(statement, successor);
			auto symbols = graph->GetEdgeData(statement, successor);

			if (statementOverlay == containerOverlay)
			{
				if (successorOverlay == containerOverlay)
				{
					// Both statement in the container overlay

					InsertEdge(subgraph, statement, successor, isBackEdge, symbols);
				}
				else
				{
					// Statement in container, successor in an overlay

					InsertEdge(subgraph, statement, successorOverlay, isBackEdge, symbols);
				}
			}
			else
			{
				if (successorOverlay == containerOverlay)
				{
					// Statement in overlay, successor in container overlay

					InsertEdge(subgraph, statementOverlay, successor, isBackEdge, symbols);
				}
				else if (statementOverlay != successorOverlay)
				{
					// Connect two separate overlays

					InsertEdge(subgraph, statementOverlay, successorOverlay, isBackEdge, symbols);
				}
			}
		}
	}

	for (const auto& childOverlay : overlay->GetChildren())
	{
		ProcessOverlay(childOverlay, containerOverlay);
	}
}

void DependencySubgraphAnalysis::Visit(DependencyOverlay *overlay)
{
	// Recursively traverse all child overlays, constructing the subgraphs for each level

	auto subgraph = new DependencySubgraph();
	overlay->SetSubgraph(subgraph);

	for (auto& childOverlay : overlay->GetChildren())
	{
		subgraph->InsertNode(childOverlay);
		childOverlay->Accept(*this);
	}

	// Add all statements to the subgraph

	for (const auto& statement : overlay->GetStatements())
	{
		subgraph->InsertNode(statement);
	}

	// Add edges connecting statements and top-level overlays

	ProcessOverlay(overlay, overlay);
}

}
