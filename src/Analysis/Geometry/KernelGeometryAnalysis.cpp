#include "Analysis/Geometry/KernelGeometryAnalysis.h"

#include "Analysis/Geometry/GeometryUtils.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Analysis {

void KernelGeometryAnalysis::Analyze(const HorseIR::Function *function)
{
	auto timeKernel_start = Utils::Chrono::Start("Kernel analysis '" + function->GetName() + "'");
	function->Accept(*this);
	Utils::Chrono::End(timeKernel_start);

	if (m_operatingGeometry == nullptr)
	{
		Utils::Logger::LogError("Unable to determine kernel geometry of empty kernel '" + function->GetName() + "'");
	}

	if (Utils::Options::Present(Utils::Options::Opt_Print_analysis))
	{
		auto string = "Kernel geometry '" + function->GetName() + "': " + ShapeUtils::ShapeString(m_operatingGeometry);
		Utils::Logger::LogInfo(string);
	}
}

bool KernelGeometryAnalysis::VisitIn(const HorseIR::Statement *statement)
{
	auto statementGeometry = m_geometryAnalysis.GetGeometry(statement);
	m_operatingGeometry = GeometryUtils::MaxGeometry(m_operatingGeometry, statementGeometry);
	return false;
}

bool KernelGeometryAnalysis::VisitIn(const HorseIR::DeclarationStatement *declarationS)
{
	// Exclude declaration statements from geometry

	return false;
}

bool KernelGeometryAnalysis::VisitIn(const HorseIR::ReturnStatement *returnS)
{
	// Exclude return statements from geometry

	return false;
}

}
