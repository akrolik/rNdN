#include "Analysis/Shape/ShapeCollector.h"

namespace Analysis {

const Shape *ShapeCollector::ShapeFromOperand(const ShapeAnalysisProperties& shapes, const HorseIR::Operand *operand)
{
	ShapeCollector collector(shapes);
	collector.Analyze(operand);
	return collector.GetShape();
}

void ShapeCollector::Analyze(const HorseIR::Operand *operand)
{
	m_shape = nullptr;
	operand->Accept(*this);
}

void ShapeCollector::Visit(const HorseIR::Identifier *identifier)
{
	m_shape = m_shapes.at(identifier->GetSymbol());
}

void ShapeCollector::Visit(const HorseIR::VectorLiteral *literal)
{
	m_shape = new VectorShape(new Shape::ConstantSize(literal->GetCount()));
}

}
