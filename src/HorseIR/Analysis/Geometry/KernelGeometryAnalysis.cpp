#include "HorseIR/Analysis/Geometry/KernelGeometryAnalysis.h"

#include "HorseIR/Analysis/Geometry/GeometryUtils.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace HorseIR {
namespace Analysis {

void KernelGeometryAnalysis::Analyze(const Function *function)
{
	auto timeKernel_start = Utils::Chrono::Start("Kernel analysis '" + function->GetName() + "'");
	function->Accept(*this);
	Utils::Chrono::End(timeKernel_start);

	if (m_operatingGeometry == nullptr)
	{
		Utils::Logger::LogError("Unable to determine kernel geometry of empty kernel '" + function->GetName() + "'");
	}

	if (Utils::Options::IsFrontend_PrintAnalysis())
	{
		auto string = "Kernel geometry '" + function->GetName() + "': " + ShapeUtils::ShapeString(m_operatingGeometry);
		Utils::Logger::LogInfo(string);
	}
}

bool KernelGeometryAnalysis::VisitIn(const Statement *statement)
{
	auto statementGeometry = m_geometryAnalysis.GetGeometry(statement);
	m_operatingGeometry = GeometryUtils::MaxGeometry(m_operatingGeometry, statementGeometry);
	return false;
}

bool KernelGeometryAnalysis::VisitIn(const DeclarationStatement *declarationS)
{
	// Exclude declaration statements from geometry

	return false;
}

bool KernelGeometryAnalysis::VisitIn(const ReturnStatement *returnS)
{
	// Exclude return statements from geometry

	return false;
}

}
}
