#include "HorseIR/Analysis/Shape/ShapeResults.h"

#include "Utils/Logger.h"

namespace HorseIR {

Shape *MethodInvocationShapes::GetShape(const Expression *expression) const
{
	if (m_expressionShapes.find(expression) == m_expressionShapes.end())
	{
		Utils::Logger::LogError("Shape not found for expression '" + expression->ToString() + "'");
	}
	return m_expressionShapes.at(expression);
}

Shape *MethodInvocationShapes::GetShape(const std::string& variable) const
{
	if (m_variableShapes.find(variable) == m_variableShapes.end())
	{
		Utils::Logger::LogError("Shape not found for variable '" + variable + "'");
	}
	return m_variableShapes.at(variable);
}

void MethodInvocationShapes::SetShape(const Expression *expression, Shape *shape)
{
	m_expressionShapes.insert({expression, shape});
}

void MethodInvocationShapes::SetShape(const std::string& variable, Shape *shape)
{
	m_variableShapes.insert({variable, shape});
}

MethodInvocationShapes *ShapeResults::GetInvocationShapes(const CallExpression *call) const
{
	if (m_invocationShapes.find(call) == m_invocationShapes.end())
	{
		Utils::Logger::LogError("Shapes for method invocation '" + call->GetMethod()->GetName() + "' not found");
	}
	return m_invocationShapes.at(call);
}

void ShapeResults::AddInvocationShapes(const CallExpression *call, MethodInvocationShapes *results)
{
	m_invocationShapes.insert({call, results});
}

}
