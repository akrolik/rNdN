#pragma once

#include <vector>

#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "Analysis/Geometry/GeometryAnalysis.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

class KernelAnalysis : public HorseIR::ConstHierarchicalVisitor
{
public:
	KernelAnalysis(const GeometryAnalysis& geometryAnalysis) : m_geometryAnalysis(geometryAnalysis) {}

	void Analyze(const HorseIR::Function *function);
	const Shape *GetOperatingGeometry() const { return m_operatingGeometry; }

	bool VisitIn(const HorseIR::Statement *statement) override;
	bool VisitIn(const HorseIR::DeclarationStatement *declarationS) override;
	bool VisitIn(const HorseIR::ReturnStatement *returnS) override;

private:
	const GeometryAnalysis& m_geometryAnalysis;

	const Shape *m_operatingGeometry = nullptr;
};

}
