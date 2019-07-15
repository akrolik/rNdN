#pragma once

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeAnalysis.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

class ShapeCollector : public HorseIR::ConstVisitor
{
public:
	static const Shape *ShapeFromOperand(const ShapeAnalysisProperties& shapes, const HorseIR::Operand *operand);
	
	ShapeCollector(const ShapeAnalysisProperties& shapes) : m_shapes(shapes) {}

	void Analyze(const HorseIR::Operand *operand);
	const Shape *GetShape() const { return m_shape; }

	void Visit(const HorseIR::Identifier *identifier) override;
	void Visit(const HorseIR::VectorLiteral *literal) override;

private:
	const ShapeAnalysisProperties& m_shapes;

	const Shape *m_shape = nullptr;
};

}
