#include "Analysis/Shape/ShapeAnalysis.h"

#include "Analysis/Shape/ShapeAnalysisHelper.h"
#include "Analysis/Shape/ShapeCollector.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Utils/Logger.h"

namespace Analysis {

void ShapeAnalysis::Visit(const HorseIR::Parameter *parameter)
{
	// Add dynamic sized shapes for all parameters

	m_currentOutSet = m_currentInSet;
	
	auto symbol = parameter->GetSymbol();
	if (m_currentOutSet.find(symbol) == m_currentOutSet.end())
	{
		m_currentOutSet[parameter->GetSymbol()] = ShapeUtils::SymbolicShapeFromType(parameter->GetType(), "param." + parameter->GetName());
	}

	m_parameterShapes[parameter] = m_currentOutSet.at(parameter->GetSymbol());
}

void ShapeAnalysis::Visit(const HorseIR::DeclarationStatement *declarationS)
{
	m_currentOutSet = m_currentInSet;

	// Set initial geometry for variable

	auto declaration = declarationS->GetDeclaration();
	m_currentOutSet[declaration->GetSymbol()] = ShapeUtils::InitialShapeFromType(declaration->GetType());
}

void ShapeAnalysis::Visit(const HorseIR::AssignStatement *assignS)
{
	// For each target, update the shape with the shape from the expression

	auto expression = assignS->GetExpression();
	auto expressionShapes = ShapeAnalysisHelper::GetShapes(m_currentInSet, expression, m_enforce);

	m_expressionShapes[expression] = expressionShapes;

	// Check the number of shapes matches the number of targets

	auto targets = assignS->GetTargets();
	if (expressionShapes.size() != targets.size())
	{
		Utils::Logger::LogError("Mismatched number of shapes for assignment. Received " + std::to_string(expressionShapes.size()) + ", expected " + std::to_string(targets.size()) + ".");
	}

	// Update map for each target symbol

	m_currentOutSet = m_currentInSet;

	unsigned int i = 0;
	for (const auto target : targets)
	{
		// Extract the target shape from the expression

		auto symbol = target->GetSymbol();
		m_currentOutSet[symbol] = expressionShapes.at(i++);
	}
}

void ShapeAnalysis::Visit(const HorseIR::ExpressionStatement *expressionS)
{
	ForwardAnalysis<ShapeAnalysisProperties>::Visit(expressionS);

	// Store the result shape for other transformations/analyses

	auto expression = expressionS->GetExpression();
	auto expressionShapes = ShapeAnalysisHelper::GetShapes(m_currentInSet, expression);

	m_expressionShapes[expression] = expressionShapes;
}

void ShapeAnalysis::Visit(const HorseIR::BlockStatement *blockS)
{
	// Visit all statements within the block and compute the sets

	ForwardAnalysis<ShapeAnalysisProperties>::Visit(blockS);

	// Kill all declarations that were part of the block

	auto symbolTable = blockS->GetSymbolTable();
	auto it = m_currentOutSet.begin();
	while (it != m_currentOutSet.end())
	{
		auto symbol = it->first;
		if (symbolTable->ContainsSymbol(symbol))
		{
			it = m_currentOutSet.erase(it);
		}
		else
		{
			++it;
		}
	}
}

void ShapeAnalysis::Visit(const HorseIR::IfStatement *ifS)
{
	ForwardAnalysis<ShapeAnalysisProperties>::Visit(ifS);

	CheckCondition(GetInSet(ifS), ifS->GetCondition());
}

void ShapeAnalysis::Visit(const HorseIR::WhileStatement *whileS)
{
	ForwardAnalysis<ShapeAnalysisProperties>::Visit(whileS);

	CheckCondition(GetInSet(whileS), whileS->GetCondition());
}

void ShapeAnalysis::Visit(const HorseIR::RepeatStatement *repeatS)
{
	ForwardAnalysis<ShapeAnalysisProperties>::Visit(repeatS);

	CheckCondition(GetInSet(repeatS), repeatS->GetCondition());
}

void ShapeAnalysis::CheckCondition(const ShapeAnalysisProperties& shapes, const HorseIR::Operand *operand)
{
	auto conditionShape = ShapeCollector::ShapeFromOperand(shapes, operand);
	if (!ShapeUtils::IsShape<VectorShape>(conditionShape))
	{
		Utils::Logger::LogError("Condition expects a scalar expression");
	}

	auto conditionSize = ShapeUtils::GetShape<VectorShape>(conditionShape)->GetSize();
	if (ShapeUtils::IsSize<Shape::ConstantSize>(conditionSize))
	{
		if (!ShapeUtils::IsScalarSize(conditionSize))
		{
			Utils::Logger::LogError("Condition expects a scalar expression");
		}
	}
}

void ShapeAnalysis::Visit(const HorseIR::ReturnStatement *returnS)
{
	ForwardAnalysis<ShapeAnalysisProperties>::Visit(returnS);

	std::vector<const Shape *> returnShapes;
	for (const auto& operand : returnS->GetOperands())
	{
		returnShapes.push_back(ShapeCollector::ShapeFromOperand(this->m_currentOutSet, operand));
	}

	if (m_returnShapes.size() == 0)
	{
		m_returnShapes = returnShapes;
	}
	else
	{
		for (auto i = 0u; i < m_returnShapes.size(); ++i)
		{
			m_returnShapes.at(i) = ShapeUtils::MergeShape(m_returnShapes.at(i), returnShapes.at(i));
		}
	}
}

ShapeAnalysis::Properties ShapeAnalysis::InitialFlow() const
{
	// Add all global variables to the initial flow set

	Properties initialFlow;
	for (const auto module : m_program->GetModules())
	{
		for (const auto content : module->GetContents())
		{
			if (auto global = dynamic_cast<const HorseIR::GlobalDeclaration *>(content))
			{
				auto declaration = global->GetDeclaration();
				initialFlow[declaration->GetSymbol()] = ShapeUtils::ShapeFromType(declaration->GetType());
			}
		}
	}
	return initialFlow;
}

ShapeAnalysis::Properties ShapeAnalysis::Merge(const Properties& s1, const Properties& s2) const
{
	// Merge the maps using a shape merge operation on each element

	Properties outSet(s1);
	for (const auto& [symbol, shape] : s2)
	{
		auto it = outSet.find(symbol);
		if (it != outSet.end())
		{
			// Merge shapes according to the rules

			outSet[symbol] = ShapeUtils::MergeShape(shape, it->second);
		}
		else
		{
			outSet.insert({symbol, shape});
		}
	}
	return outSet;
}

}
