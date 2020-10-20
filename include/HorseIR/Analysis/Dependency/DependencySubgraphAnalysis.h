#pragma once

#include <unordered_set>

#include "HorseIR/Analysis/Dependency/Overlay/DependencyOverlayVisitor.h"

#include "HorseIR/Analysis/Dependency/DependencyGraph.h"

namespace HorseIR {
namespace Analysis {

class DependencySubgraphAnalysis : public DependencyOverlayVisitor
{
public:
	// Inputs/outputs

	void Analyze(DependencyOverlay *overlay);

	// Dependency overlay visitor

	void Visit(DependencyOverlay *overlay) override;

private:
	const DependencyOverlay *GetScopedOverlay(const DependencyOverlay *containerOverlay, const Statement *statement) const;
	void ProcessOverlay(const DependencyOverlay *overlay, const DependencyOverlay *containerOverlay);
};

}
}
