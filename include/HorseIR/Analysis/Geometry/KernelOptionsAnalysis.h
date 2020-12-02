#pragma once

#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "Frontend/Codegen/InputOptions.h"

#include "HorseIR/Analysis/Shape/ShapeAnalysis.h"
#include "HorseIR/Tree/Tree.h"

namespace HorseIR {
namespace Analysis {

class KernelOptionsAnalysis : public ConstHierarchicalVisitor
{
public:
	KernelOptionsAnalysis(const ShapeAnalysis& shapeAnalysis) : m_shapeAnalysis(shapeAnalysis) {}

	void Analyze(const Function *function);

	Frontend::Codegen::InputOptions *GetInputOptions() const { return m_inputOptions; }

	bool IsInOrder(const Function *function) const;

	bool VisitIn(const Parameter *parameter) override;
	bool VisitIn(const VariableDeclaration *declaration) override;

private:
	std::uint32_t GetAverageCellSize(const Analysis::ListShape *shape) const;

	const ShapeAnalysis& m_shapeAnalysis;

	Frontend::Codegen::InputOptions *m_inputOptions = nullptr;
};

}
}
