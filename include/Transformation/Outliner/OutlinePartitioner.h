#pragma once

#include <queue>
#include <unordered_set>
#include <variant>
#include <vector>

#include "Analysis/Compatibility/Overlay/CompatibilityOverlayConstVisitor.h"

#include "Analysis/Compatibility/Geometry/Geometry.h"
#include "Analysis/Compatibility/Geometry/GeometryAnalysis.h"
#include "Analysis/Compatibility/Overlay/CompatibilityOverlay.h"

namespace Transformation {

class OutlinePartitioner : public Analysis::CompatibilityOverlayConstVisitor
{
public:
	OutlinePartitioner(const Analysis::GeometryAnalysis& geometryAnalysis) : m_geometryAnalysis(geometryAnalysis) {}

	Analysis::CompatibilityOverlay *Partition(const Analysis::CompatibilityOverlay *overlay);

	void Visit(const Analysis::CompatibilityOverlay *overlay) override;

	void Visit(const Analysis::FunctionCompatibilityOverlay *overlay) override;
	void Visit(const Analysis::IfCompatibilityOverlay *overlay) override;
	void Visit(const Analysis::WhileCompatibilityOverlay *overlay) override;
	void Visit(const Analysis::RepeatCompatibilityOverlay *overlay) override;

private:
	using NodeType = std::variant<const HorseIR::Statement *, Analysis::CompatibilityOverlay *>;

	struct PartitionContext
	{
		PartitionContext(std::unordered_set<NodeType>& visited, std::queue<NodeType>& queue) : visited(visited), queue(queue) {}

		std::unordered_set<NodeType>& visited;
		std::queue<NodeType>& queue;
	};

	Analysis::CompatibilityOverlay *GetChildOverlay(const std::vector<Analysis::CompatibilityOverlay *>& overlays, const HorseIR::Statement *statement) const;

	void ProcessEdges(PartitionContext& context, const Analysis::CompatibilityOverlay *overlay, const HorseIR::Statement *statement, const std::unordered_set<const HorseIR::Statement *>& destinations, bool flipped);
	void ProcessOverlayEdges(PartitionContext& context, const Analysis::CompatibilityOverlay *entry, const Analysis::CompatibilityOverlay *overlay);

	template<typename T>
	void VisitLoop(const T *overlay);

	std::vector<Analysis::CompatibilityOverlay *> m_currentOverlays;

	const Analysis::GeometryAnalysis& m_geometryAnalysis;
	std::unordered_map<const Analysis::CompatibilityOverlay *, const Analysis::Geometry *> m_overlayGeometries;
};

}
