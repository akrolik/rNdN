#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeAnalysis.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {
namespace Analysis {

class ShapeCollector : public ConstVisitor
{
public:
	static const Shape *ShapeFromOperand(const ShapeAnalysisProperties& shapes, const Operand *operand);
	
	ShapeCollector(const ShapeAnalysisProperties& shapes) : m_shapes(shapes) {}

	void Analyze(const Operand *operand);
	const Shape *GetShape() const { return m_shape; }

	void Visit(const Identifier *identifier) override;
	void Visit(const VectorLiteral *literal) override;

private:
	const ShapeAnalysisProperties& m_shapes;

	const Shape *m_shape = nullptr;
};

}
}
