#pragma once

#include <vector>

#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "HorseIR/Analysis/Geometry/GeometryAnalysis.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {
namespace Analysis {

class KernelGeometryAnalysis : public ConstHierarchicalVisitor
{
public:
	inline const static std::string Name = "Kernel geometry analysis";
	inline const static std::string ShortName = "kernelgeom";

	// Public API

	KernelGeometryAnalysis(const GeometryAnalysis& geometryAnalysis) : m_geometryAnalysis(geometryAnalysis) {}

	void Analyze(const Function *function);
	const Shape *GetOperatingGeometry() const { return m_operatingGeometry; }

	// Visitors

	bool VisitIn(const Statement *statement) override;
	bool VisitIn(const DeclarationStatement *declarationS) override;
	bool VisitIn(const ReturnStatement *returnS) override;

private:
	const GeometryAnalysis& m_geometryAnalysis;

	const Shape *m_operatingGeometry = nullptr;
};

}
}
