#pragma once

#include <unordered_set>

#include "Analysis/Dependency/Overlay/DependencyOverlayVisitor.h"

#include "Analysis/Dependency/DependencyGraph.h"

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
};

}
