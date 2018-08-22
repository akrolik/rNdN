#pragma once

#include <stack>
#include <tuple>
#include <vector>

#include "HorseIR/Traversal/ConstForwardTraversal.h"

#include "HorseIR/Analysis/Shape.h"
#include "HorseIR/Analysis/ShapeResults.h"
#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class ShapeAnalysis : public ConstForwardTraversal
{
public:
	void Analyze(Method *method);

	void Visit(const Parameter *parameter) override;
	void Visit(const AssignStatement *assign) override;
	void Visit(const ReturnStatement *ret) override;

	void Visit(const CallExpression *call) override;
	void Visit(const CastExpression *cast) override;
	void Visit(const Identifier *identifier) override;

	void Visit(const BoolLiteral *literal);
	void Visit(const Int8Literal *literal);
	void Visit(const Int16Literal *literal);
	void Visit(const Int32Literal *literal);
	void Visit(const Int64Literal *literal);
	void Visit(const Float32Literal *literal);
	void Visit(const Float64Literal *literal);
	void Visit(const StringLiteral *literal);
	void Visit(const SymbolLiteral *literal);
	void Visit(const DateLiteral *literal);
	void Visit(const FunctionLiteral *literal);

private:
	[[noreturn]] void ShapeError(const MethodDeclaration *method, const std::vector<Expression *>& arguments);

	Shape *AnalyzeCall(const MethodDeclaration *method, const std::vector<Expression *>& arguments);
	Shape *AnalyzeCall(const Method *method, const std::vector<Expression *>& arguments);
	Shape *AnalyzeCall(const BuiltinMethod *method, const std::vector<Expression *>& arguments);

	Shape *GetShape(const Expression *expression);
	Shape *GetShape(const std::string& variable);

	void SetShape(const Expression *expression, Shape *shape);
	void SetShape(const std::string& variable, Shape *shape);

	ShapeResults *m_results = new ShapeResults();
	std::stack<std::tuple<const CallExpression *, MethodInvocationShapes *>> m_shapes;

	const CallExpression *m_context = nullptr;
	Shape *m_contextResult = nullptr;
};

}
