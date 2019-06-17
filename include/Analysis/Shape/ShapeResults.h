#pragma once

#include <string>
#include <unordered_map>

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Tree/Expressions/Expression.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"

namespace HorseIR {

class MethodInvocationShapes
{
public:
	MethodInvocationShapes(const CallExpression *call) : m_call(call) {}

	Shape *GetShape(const Expression *expression) const;
	Shape *GetShape(const std::string& variable) const;

	void SetShape(const Expression *expression, Shape *shape);
	void SetShape(const std::string& variable, Shape *shape);

	Shape *GetReturnShape() const { return m_returnShape; }
	void SetReturnShape(Shape *shape) { m_returnShape = shape; }

private:
	const CallExpression *m_call = nullptr;

	std::unordered_map<const Expression *, Shape *> m_expressionShapes;
	std::unordered_map<std::string, Shape *> m_variableShapes;

	Shape *m_returnShape = nullptr;
};

class ShapeResults
{
public:
	MethodInvocationShapes *GetInvocationShapes(const CallExpression *call) const;
	void AddInvocationShapes(const CallExpression *call, MethodInvocationShapes *results);
	
private:
	std::unordered_map<const CallExpression *, MethodInvocationShapes *> m_invocationShapes;
};

}
