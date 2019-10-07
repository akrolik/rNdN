#include "Analysis/Geometry/KernelAnalysis.h"

#include "Analysis/Geometry/GeometryUtils.h"

namespace Analysis {

void KernelAnalysis::Analyze(const HorseIR::Function *function)
{
	function->Accept(*this);

	switch (m_maxShape->GetKind())
	{
		case Shape::Kind::Vector:
		{
			const auto shape = ShapeUtils::GetShape<VectorShape>(m_maxShape);
			if (const auto size = ShapeUtils::GetSize<Shape::ConstantSize>(shape->GetSize()))
			{
				m_threadGeometry = new ThreadGeometry(ThreadGeometry::Kind::Vector, size->GetValue());
			}
			else
			{
				Utils::Logger::LogError("Vector geometry must have constant size for kernel execution");
			}
			break;
		}
		case Shape::Kind::List:
		{
			const auto shape = ShapeUtils::GetShape<ListShape>(m_maxShape);
			if (const auto size = ShapeUtils::GetSize<Shape::ConstantSize>(shape->GetListSize()))
			{
				m_threadGeometry = new ThreadGeometry(ThreadGeometry::Kind::List, size->GetValue());
			}
			else
			{
				Utils::Logger::LogError("List geometry must have constant cell count for kernel execution");
			}
			break;
		}
		default:
		{
			Utils::Logger::LogError("Unsupport thread geometry kind");
		}
	}
}

bool KernelAnalysis::VisitIn(const HorseIR::Statement *statement)
{
	auto statementGeometry = m_geometryAnalysis.GetGeometry(statement);
	m_maxShape = GeometryUtils::MaxGeometry(m_maxShape, statementGeometry);
	return false;
}

bool KernelAnalysis::VisitIn(const HorseIR::DeclarationStatement *declarationS)
{
	//TODO: Handle flexible geometry
	return false;
}

}
