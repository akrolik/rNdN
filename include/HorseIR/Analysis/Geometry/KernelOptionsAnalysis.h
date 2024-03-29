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
	inline const static std::string Name = "Kernel options analysis";
	inline const static std::string ShortName = "kernelopts";

	// Public API

	KernelOptionsAnalysis(const ShapeAnalysis& shapeAnalysis) : m_shapeAnalysis(shapeAnalysis) {}

	Frontend::Codegen::InputOptions *Analyze(const Function *function);

	bool IsInOrder(const Function *function) const;

	// Visitors

	bool VisitIn(const Parameter *parameter) override;
	bool VisitIn(const VariableDeclaration *declaration) override;

private:
	std::uint32_t GetAverageCellSize(const Analysis::ListShape *shape) const;

	const ShapeAnalysis& m_shapeAnalysis;

	Frontend::Codegen::InputOptions *m_inputOptions = nullptr;
};

}
}
