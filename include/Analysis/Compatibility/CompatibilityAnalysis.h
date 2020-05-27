#pragma once

#include <vector>
#include <unordered_map>

#include "Analysis/Dependency/Overlay/DependencyOverlayConstVisitor.h"

#include "Analysis/Dependency/DependencyGraph.h"
#include "Analysis/Geometry/GeometryAnalysis.h"
#include "Analysis/Shape/Shape.h"

#include "Utils/Variant.h"

namespace Analysis {

class CompatibilityAnalysis : public DependencyOverlayConstVisitor
{
public:
	CompatibilityAnalysis(const GeometryAnalysis& geometryAnalysis) : m_geometryAnalysis(geometryAnalysis) {}

	// Analysis input/output

	void Analyze(const FunctionDependencyOverlay *overlay);

	FunctionDependencyOverlay *GetOverlay() const { return m_functionOverlay; }

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

	bool IsIterable(const DependencyOverlay *overlay) const;

	// Utilities for handling overlay construction

	FunctionDependencyOverlay *m_functionOverlay = nullptr;
	std::vector<DependencyOverlay *> m_currentOverlays;
	std::unordered_map<DependencySubgraphNode, DependencyOverlay *> m_kernelMap;
	std::unordered_map<const DependencyOverlay *, DependencyOverlay *> m_overlayMap;

	// Geometry analysis for statements and map for overlays

	const GeometryAnalysis& m_geometryAnalysis;
	std::unordered_map<const DependencyOverlay *, const Shape *> m_overlayGeometries;

	const Shape *GetGeometry(const DependencySubgraphNode& node) const;
	const Shape *BuildGeometry(const DependencyOverlay *overlay) const;
	bool IsCompatible(const Shape *source, const Shape *destination, bool allowCompression = false) const;
	bool IsCompatible(const Shape::Size *source, const Shape::Size *destination, bool allowCompression = false) const;

	// Optimization

	void Optimize(DependencyOverlay *parentOverlay);
	DependencyOverlay *MergeOverlays(DependencySubgraph::OrderingContext& context, const std::unordered_set<DependencySubgraphNode>& processedNodes, const DependencyOverlay *overlay1, const DependencyOverlay *overlay2);
	void MoveOverlay(DependencySubgraph::OrderingContext& context, DependencyOverlay *merged, const DependencyOverlay *source);
};

}
