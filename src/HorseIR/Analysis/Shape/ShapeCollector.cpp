#include "HorseIR/Analysis/Shape/ShapeCollector.h"

namespace HorseIR {
namespace Analysis {

const Shape *ShapeCollector::ShapeFromOperand(const ShapeAnalysisProperties& shapes, const Operand *operand)
{
	ShapeCollector collector(shapes);
	collector.Analyze(operand);
	return collector.GetShape();
}

void ShapeCollector::Analyze(const Operand *operand)
{
	m_shape = nullptr;
	operand->Accept(*this);
}

void ShapeCollector::Visit(const Identifier *identifier)
{
	m_shape = m_shapes.first.at(identifier->GetSymbol());
}

void ShapeCollector::Visit(const VectorLiteral *literal)
{
	m_shape = new VectorShape(new Shape::ConstantSize(literal->GetCount()));
}

}
}
