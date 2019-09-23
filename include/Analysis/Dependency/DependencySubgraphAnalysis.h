#pragma once

#include <unordered_set>

#include "Analysis/Dependency/Overlay/DependencyOverlayVisitor.h"

#include "Analysis/Dependency/DependencyGraph.h"
#include "Analysis/Dependency/DependencySubgraph.h"

namespace Analysis {

class DependencySubgraphAnalysis : public DependencyOverlayVisitor
{
public:
	// Inputs/outputs

	void Analyze(DependencyOverlay *overlay);

	// Dependency overlay visitor

	void Visit(DependencyOverlay *overlay) override;

private:
	const DependencyOverlay *GetScopedOverlay(const DependencyOverlay *containerOverlay, const HorseIR::Statement *statement) const;

	void ProcessOverlay(const DependencyOverlay *overlay, const DependencyOverlay *containerOverlay);
	static void InsertEdge(DependencySubgraph *subgraph, const DependencySubgraphNode& source, const DependencySubgraphNode& destination, bool isBackEdge, const std::unordered_set<const HorseIR::SymbolTable::Symbol *>& symbols);
};

}
