#pragma once

#include <string>
#include <unordered_map>

#include "HorseIR/Analysis/Shape.h"
#include "HorseIR/Tree/Expressions/Expression.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"

#include "Utils/Logger.h"

namespace HorseIR {

class MethodInvocationShapes
{
public:
	Shape *GetShape(const Expression *expression)
	{
		if (m_expressionShapes.find(expression) == m_expressionShapes.end())
		{
			Utils::Logger::LogError("Shape not found for expression '" + expression->ToString() + "'");
		}
		return m_expressionShapes.at(expression);
	}

	Shape *GetShape(const std::string& variable)
	{
		if (m_variableShapes.find(variable) == m_variableShapes.end())
		{
			Utils::Logger::LogError("Shape not found for variable '" + variable + "'");
		}
		return m_variableShapes.at(variable);
	}

	void SetShape(const Expression *expression, Shape *shape)
	{
		m_expressionShapes.insert({expression, shape});
	}

	void SetShape(const std::string& variable, Shape *shape)
	{
		m_variableShapes.insert({variable, shape});
	}

private:
	std::unordered_map<const Expression *, Shape *> m_expressionShapes;
	std::unordered_map<std::string, Shape *> m_variableShapes;
};

class ShapeResults
{
public:
	void AddInvocationShapes(const CallExpression *call, MethodInvocationShapes *results)
	{
		m_invocationShapes.insert({call, results});
	}
	
private:
	std::unordered_map<const CallExpression *, MethodInvocationShapes *> m_invocationShapes;
};

}
