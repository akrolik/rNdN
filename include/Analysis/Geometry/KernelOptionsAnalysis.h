#pragma once

#include "Analysis/Shape/ShapeAnalysis.h"

#include "Codegen/InputOptions.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

namespace Analysis {

class KernelOptionsAnalysis : public HorseIR::ConstHierarchicalVisitor
{
public:
	KernelOptionsAnalysis(const ShapeAnalysis& shapeAnalysis) : m_shapeAnalysis(shapeAnalysis) {}

	void Analyze(const HorseIR::Function *function);

	Codegen::InputOptions *GetInputOptions() const { return m_inputOptions; }

	bool IsInOrder(const HorseIR::Function *function) const;

	bool VisitIn(const HorseIR::Parameter *parameter) override;
	bool VisitIn(const HorseIR::VariableDeclaration *declaration) override;

private:
	std::uint32_t GetAverageCellSize(const Analysis::ListShape *shape) const;

	const ShapeAnalysis& m_shapeAnalysis;

	Codegen::InputOptions *m_inputOptions = nullptr;
};

}
