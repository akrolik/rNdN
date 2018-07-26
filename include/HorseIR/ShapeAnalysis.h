#pragma once

#include <unordered_map>

#include "HorseIR/Traversal/ForwardTraversal.h"

#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

struct Shape
{
	enum class Kind {
		Vector,
		List,
		Table
	};

	Shape(Kind k, unsigned int s) : kind(k), size(s) {}

	Kind kind;
	unsigned int size;
};

using ShapeMap = std::unordered_map<std::string, Shape *>;
using ExpressionMap = std::unordered_map<Expression *, Shape *>;

class ShapeAnalysis : public ForwardTraversal
{
public:
	void SetInputShape(const Parameter *parameter, Shape *shape);

	void Analyze(Method *method);
	Shape *GetShape(const std::string& identifier) const;

	void Visit(Parameter *parameter) override;
	void Visit(AssignStatement *assign) override;

	void Visit(CallExpression *call);
	// void Visit(CastExpression *cast);
	// void Visit(Identifier *identifier);
	// void Visit(Literal<int64_t> *literal);
	// void Visit(Literal<double> *literal);
	// void Visit(Literal<std::string> *literal);
	// void Visit(Symbol *symbol);

private:
	Shape *GetExpressionShape(Expression *expression);

	ShapeMap m_identifierMap;
	ExpressionMap m_expressionMap;
};

}
