#pragma once

#include <vector>

#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "Analysis/Geometry/GeometryAnalysis.h"
#include "Analysis/Geometry/ThreadGeometry.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

class KernelAnalysis : public HorseIR::ConstHierarchicalVisitor
{
public:
	KernelAnalysis(const GeometryAnalysis& geometryAnalysis) : m_geometryAnalysis(geometryAnalysis) {}

	void Analyze(const HorseIR::Function *function);
	const ThreadGeometry *GetThreadGeometry() const { return m_threadGeometry; }

	bool VisitIn(const HorseIR::Statement *statement) override;
	bool VisitIn(const HorseIR::DeclarationStatement *declarationS) override;

private:
	const GeometryAnalysis& m_geometryAnalysis;

	const ThreadGeometry *m_threadGeometry = nullptr;
	const Shape *m_maxShape = nullptr;
};

}
