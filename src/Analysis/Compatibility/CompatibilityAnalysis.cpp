#include "Analysis/Compatibility/CompatibilityAnalysis.h"

#include "Analysis/Helpers/GPUAnalysisHelper.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

void CompatibilityAnalysis::Analyze(const HorseIR::Function *function)
{
	function->Accept(*this);
}

bool CompatibilityAnalysis::VisitIn(const HorseIR::Function *function)
{
	auto functionOverlay = new FunctionCompatibilityOverlay(function, m_graph);
	m_graphOverlay = new CompatibilityOverlay(m_graph, functionOverlay);
	return true;
}

void CompatibilityAnalysis::VisitOut(const HorseIR::Function *function)
{
	m_graphOverlay = m_graphOverlay->GetParent();
}

bool CompatibilityAnalysis::VisitIn(const HorseIR::Statement *statement)
{
	// Add statement to graph, and set as current for identifier dependencies and check GPU compatibility

	GPUAnalysisHelper gpuHelper;
	gpuHelper.Analyze(statement);

	m_graph->InsertNode(statement, gpuHelper.IsCapable(), gpuHelper.IsSynchronized());
	m_graphOverlay->InsertStatement(statement);

	m_currentStatement = statement;
	return true;
}

void CompatibilityAnalysis::VisitOut(const HorseIR::Statement *statement)
{
	m_currentStatement = nullptr;
}

bool CompatibilityAnalysis::VisitIn(const HorseIR::AssignStatement *assignS)
{
	// Determine if the operation is GPU capable

	GPUAnalysisHelper gpuHelper;
	gpuHelper.Analyze(assignS);

	// Only traverse the expression, targets do not introduce dependencies

	m_graph->InsertNode(assignS, gpuHelper.IsCapable(), gpuHelper.IsSynchronized());
	m_graphOverlay->InsertStatement(assignS);

	m_currentStatement = assignS;

	assignS->GetExpression()->Accept(*this);

	m_currentStatement = nullptr;
	return false;
}

bool CompatibilityAnalysis::VisitIn(const HorseIR::ExpressionStatement *expressionS)
{
	// Determine if the operation is GPU capable

	GPUAnalysisHelper gpuHelper;
	gpuHelper.Analyze(expressionS);

	// Insert a new node into the graph and traverse the expression

	m_graph->InsertNode(expressionS, gpuHelper.IsCapable(), gpuHelper.IsSynchronized());
	m_graphOverlay->InsertStatement(expressionS);

	m_currentStatement = expressionS;
	return true;
}

template<typename T>
void CompatibilityAnalysis::VisitCompoundStatement(const typename T::NodeType *statement)
{
	// Add the statement to the graph, all compound statements are GPU capable and unsynchronized

	m_graph->InsertNode(statement, true, false);
	m_currentStatement = statement;

	// Organize the contents in an overlay

	m_graphOverlay = new T(statement, m_graph, m_graphOverlay);
	m_graphOverlay->InsertStatement(statement);
}

bool CompatibilityAnalysis::VisitIn(const HorseIR::IfStatement *ifS)
{
	VisitCompoundStatement<IfCompatibilityOverlay>(ifS);
	return true;
}

bool CompatibilityAnalysis::VisitIn(const HorseIR::WhileStatement *whileS)
{
	VisitCompoundStatement<WhileCompatibilityOverlay>(whileS);
	return true;
}

bool CompatibilityAnalysis::VisitIn(const HorseIR::RepeatStatement *repeatS)
{
	VisitCompoundStatement<RepeatCompatibilityOverlay>(repeatS);
	return true;
}

void CompatibilityAnalysis::VisitOut(const HorseIR::IfStatement *ifS)
{
	m_graphOverlay = m_graphOverlay->GetParent();
	m_currentStatement = nullptr;
}

void CompatibilityAnalysis::VisitOut(const HorseIR::WhileStatement *whileS)
{
	m_graphOverlay = m_graphOverlay->GetParent();
	m_currentStatement = nullptr;
}

void CompatibilityAnalysis::VisitOut(const HorseIR::RepeatStatement *repeatS)
{
	m_graphOverlay = m_graphOverlay->GetParent();
	m_currentStatement = nullptr;
}

bool CompatibilityAnalysis::VisitIn(const HorseIR::BlockStatement *blockS)
{
	// Keep block statement organized in the overlay

	m_graphOverlay = new CompatibilityOverlay(m_graph, m_graphOverlay);
	return true;
}

void CompatibilityAnalysis::VisitOut(const HorseIR::BlockStatement *blockS)
{
	m_graphOverlay = m_graphOverlay->GetParent();
}

bool CompatibilityAnalysis::VisitIn(const HorseIR::FunctionLiteral *literal)
{
	// Skip function literal identifier

	return false;
}

bool CompatibilityAnalysis::VisitIn(const HorseIR::Identifier *identifier)
{
	// Add dependency for each definition, tagging with the symbols used

	auto symbol = identifier->GetSymbol();
	for (const auto& definition : m_useDefChains.GetDefinitions(identifier))
	{
		// Check for edge properties, back edge and compatibility

		bool isBackEdge = (m_graph->ContainsNode(definition) == false || definition == m_currentStatement);
		bool isCompatible = false;

		if (m_graph->IsGPUNode(definition) && m_graph->IsGPUNode(m_currentStatement))
		{
			GPUAnalysisHelper gpuHelper;
			gpuHelper.Analyze(definition);

			// For an edge to be compatible, the previous operation must not be synchronized

			if (!gpuHelper.IsSynchronized())
			{
				auto sourceGeometry = m_geometryAnalysis.GetGeometry(definition);
				auto destinationGeometry = m_geometryAnalysis.GetGeometry(m_currentStatement);
				isCompatible = IsCompatible(sourceGeometry, destinationGeometry);
			}
		}

		m_graph->InsertEdge(definition, m_currentStatement, symbol, isBackEdge, isCompatible);
	}
	return true;
}

//TODO: Review this function
bool CompatibilityAnalysis::IsCompatible(const Geometry *source, const Geometry *destination)
{
	if (*source == *destination)
	{
		return true;
	}

	// Check for compression compatibility

	if (source->GetKind() != Geometry::Kind::Shape || destination->GetKind() != Geometry::Kind::Shape)
	{
		return false;
	}

	auto sourceShape = static_cast<const ShapeGeometry *>(source)->GetShape();
	auto destinationShape = static_cast<const ShapeGeometry *>(destination)->GetShape();

	if (!ShapeUtils::IsShape<VectorShape>(sourceShape) || !ShapeUtils::IsShape<VectorShape>(destinationShape))
	{
		return false;
	}

	auto sourceSize = ShapeUtils::GetShape<VectorShape>(sourceShape)->GetSize();
	auto destinationSize = ShapeUtils::GetShape<VectorShape>(destinationShape)->GetSize();

	if (!ShapeUtils::IsSize<Shape::CompressedSize>(destinationSize))
	{
		return false;
	}

	auto unmaskedSize = ShapeUtils::GetSize<Shape::CompressedSize>(destinationSize)->GetSize();
	return (*sourceSize == *unmaskedSize);
}

}
