#pragma once

#include <unordered_map>
#include <vector>

#include "HorseIR/Traversal/ConstVisitor.h"

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeAnalysis.h"

namespace Analysis {

class ShapeAnalysisHelper : public HorseIR::ConstVisitor
{
public:
	static std::vector<const Shape *> GetShapes(const ShapeAnalysis::Properties& properties, const HorseIR::Expression *expression);

	ShapeAnalysisHelper(const ShapeAnalysis::Properties& properties) : m_properties(properties) {}

	void Visit(const HorseIR::CallExpression *call) override;
	void Visit(const HorseIR::CastExpression *cast) override;
	void Visit(const HorseIR::Identifier *identifier) override;
	void Visit(const HorseIR::VectorLiteral *literal) override;
	
private:
	std::vector<const Shape *> AnalyzeCall(const HorseIR::FunctionDeclaration *function, const std::vector<HorseIR::Operand *>& arguments);
	std::vector<const Shape *> AnalyzeCall(const HorseIR::Function *function, const std::vector<HorseIR::Operand *>& arguments);
	std::vector<const Shape *> AnalyzeCall(const HorseIR::BuiltinFunction *function, const std::vector<HorseIR::Operand *>& arguments);

	void AddScalarConstraint(const Shape::Size *size);
	void AddBinaryConstraint(const Shape::Size *size1, const Shape::Size *size2);
	void AddEqualityConstraint(const Shape::Size *size1, const Shape::Size *size2);

	[[noreturn]] void ShapeError(const HorseIR::FunctionDeclaration *method, const std::vector<HorseIR::Operand *>& arguments) const;

	const Shape *GetShape(const HorseIR::Operand *operand) const;
	void SetShape(const HorseIR::Operand *operand, const Shape *shape);

	const std::vector<const Shape *>& GetShapes(const HorseIR::Expression *expression) const;
	void SetShapes(const HorseIR::Expression *expression, const std::vector<const Shape *>& shapes);
	
	const HorseIR::CallExpression *m_call = nullptr;
	std::unordered_map<const HorseIR::Expression *, std::vector<const Shape *>> m_shapes;
	
	const ShapeAnalysis::Properties& m_properties;
};

}
