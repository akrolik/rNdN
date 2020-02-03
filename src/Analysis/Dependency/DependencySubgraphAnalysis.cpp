#include "Analysis/Dependency/DependencySubgraphAnalysis.h"

#include "Analysis/Dependency/Overlay/DependencyOverlay.h"

#include "HorseIR/Tree/Tree.h"

#include "Utils/Chrono.h"

namespace Analysis {

void DependencySubgraphAnalysis::Analyze(DependencyOverlay *overlay)
{
	auto timeDependencies_start = Utils::Chrono::Start("Dependency subgraph analysis '" + std::string(overlay->GetName()) + "'");
	overlay->Accept(*this);
	Utils::Chrono::End(timeDependencies_start);
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


			// Add edge to the subgraph with the symbol data and properties

			auto symbols = graph->GetEdgeData(statement, successor);
			auto isBackEdge = graph->IsBackEdge(statement, successor);
			auto isSynchronized = graph->IsSynchronizedEdge(statement, successor);

			if (statementOverlay == containerOverlay)
			{
				if (successorOverlay == containerOverlay)
				{
					// Both statement in the container overlay

					subgraph->InsertEdge(statement, successor, symbols, isBackEdge, isSynchronized);
				}
				else
				{
					// Statement in container, successor in an overlay

					subgraph->InsertEdge(statement, successorOverlay, symbols, isBackEdge, isSynchronized);
				}
			}
			else
			{
				if (successorOverlay == containerOverlay)
				{
					// Statement in overlay, successor in container overlay

					subgraph->InsertEdge(statementOverlay, successor, symbols, isBackEdge, isSynchronized);
				}
				else if (statementOverlay != successorOverlay)
				{
					// Connect two separate overlays

					subgraph->InsertEdge(statementOverlay, successorOverlay, symbols, isBackEdge, isSynchronized);
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
		subgraph->InsertNode(childOverlay, false, false);
		childOverlay->Accept(*this);
	}

	// Add all statements to the subgraph

	auto graph = overlay->GetGraph();
	for (const auto& statement : overlay->GetStatements())
	{
		subgraph->InsertNode(statement, graph->IsGPUNode(statement), graph->IsGPULibraryNode(statement));
	}

	// Add edges connecting statements and top-level overlays

	ProcessOverlay(overlay, overlay);
}

}
