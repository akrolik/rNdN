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
	// Entry functions for utility

	static std::vector<const Shape *> GetShapes(const ShapeAnalysis::Properties& properties, const HorseIR::Expression *expression);

	ShapeAnalysisHelper(const ShapeAnalysis::Properties& properties) : m_properties(properties) {}

	// Expressions

	void Visit(const HorseIR::CallExpression *call) override;
	void Visit(const HorseIR::CastExpression *cast) override;
	void Visit(const HorseIR::Identifier *identifier) override;
	void Visit(const HorseIR::VectorLiteral *literal) override;
	
private:
	// Function call visitors

	std::vector<const Shape *> AnalyzeCall(const HorseIR::FunctionDeclaration *function, const std::vector<const Shape *>& argumentShapes, const std::vector<HorseIR::Operand *>& arguments);
	std::vector<const Shape *> AnalyzeCall(const HorseIR::Function *function, const std::vector<const Shape *>& argumentShapes, const std::vector<HorseIR::Operand *>& arguments);
	std::vector<const Shape *> AnalyzeCall(const HorseIR::BuiltinFunction *function, const std::vector<const Shape *>& argumentShapes, const std::vector<HorseIR::Operand *>& arguments);

	// Static checks for sizes

	bool CheckStaticScalar(const Shape::Size *size, bool enforce = false) const;
	bool CheckStaticEquality(const Shape::Size *size1, const Shape::Size *size2, bool enforce = false) const;
	bool CheckStaticTabular(const ListShape *listShape, bool enforce = false) const;

	// Checks for values

	bool HasConstantArgument(const std::vector<HorseIR::Operand *>& arguments, unsigned int index) const;

	// Utility error function

	[[noreturn]] void ShapeError(const HorseIR::FunctionDeclaration *method, const std::vector<const Shape *>& argumentShapes) const;

	// Shape utilities for propagation

	const Shape *GetShape(const HorseIR::Operand *operand) const;
	void SetShape(const HorseIR::Operand *operand, const Shape *shape);

	const std::vector<const Shape *>& GetShapes(const HorseIR::Expression *expression) const;
	void SetShapes(const HorseIR::Expression *expression, const std::vector<const Shape *>& shapes);
	
	const HorseIR::CallExpression *m_call = nullptr;
	std::unordered_map<const HorseIR::Expression *, std::vector<const Shape *>> m_shapes;
	
	const ShapeAnalysis::Properties& m_properties;
};

}
