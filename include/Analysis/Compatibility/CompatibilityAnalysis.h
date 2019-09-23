#pragma once

#include <vector>
#include <unordered_map>

#include "Analysis/Dependency/Overlay/DependencyOverlayConstVisitor.h"

#include "Analysis/Compatibility/Geometry/Geometry.h"
#include "Analysis/Compatibility/Geometry/GeometryAnalysis.h"

#include "Analysis/Dependency/DependencyGraph.h"
#include "Analysis/Dependency/DependencySubgraph.h"

#include "Analysis/Helpers/GPUAnalysisHelper.h"

#include "Utils/Variant.h"

namespace Analysis {

class CompatibilityAnalysis : public DependencyOverlayConstVisitor
{
public:
	CompatibilityAnalysis(const GeometryAnalysis& geometryAnalysis) : m_geometryAnalysis(geometryAnalysis) {}

	// Analysis input/output

	void Analyze(const DependencyOverlay *overlay);

	DependencyOverlay *GetOverlay() const { return m_currentOverlays.at(0); }

	// Overlay visitors

	void Visit(const DependencyOverlay *overlay) override;

	void Visit(const FunctionDependencyOverlay *overlay) override;
	void Visit(const IfDependencyOverlay *overlay) override;
	void Visit(const WhileDependencyOverlay *overlay) override;
	void Visit(const RepeatDependencyOverlay *overlay) override;

	template<typename T>
	void VisitLoop(const T *overlay);

private:
	// Successor compatibility checking

	DependencyOverlay *GetKernelOverlay(const DependencySubgraph *subgraph, const DependencySubgraphNode& node, DependencyOverlay *parentOverlay) const;
	DependencyOverlay *GetSuccessorsKernelOverlay(const DependencySubgraph *subgraph, const DependencySubgraphNode& node) const;

	bool IsSynchronized(const DependencySubgraphNode& node) const;
	bool BuildSynchronized(const DependencyOverlay *overlay) const;

	bool IsIterable(const DependencyOverlay *overlay) const;

	// Utility for handling overlay construction

	mutable GPUAnalysisHelper m_gpuHelper;

	std::vector<DependencyOverlay *> m_currentOverlays;
	std::unordered_map<DependencySubgraphNode, DependencyOverlay *> m_kernelMap;
	std::unordered_map<const DependencyOverlay *, DependencyOverlay *> m_overlayMap;

	// Geometry analysis for statements and map for overlays

	const GeometryAnalysis& m_geometryAnalysis;
	std::unordered_map<const DependencyOverlay *, const Geometry *> m_overlayGeometries;

	const Geometry *GetGeometry(const DependencySubgraphNode& node) const;
	const Geometry *BuildGeometry(const DependencyOverlay *overlay) const;
	bool IsCompatible(const Geometry *source, const Geometry *destination) const;
};

}
