#include "Analysis/Geometry/KernelAnalysis.h"

#include "Analysis/Geometry/GeometryUtils.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Analysis {

void KernelAnalysis::Analyze(const HorseIR::Function *function)
{
	auto timeKernel_start = Utils::Chrono::Start();
	function->Accept(*this);
	auto timeKernel = Utils::Chrono::End(timeKernel_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_analysis))
	{
		auto string = "Kernel geometry: " + ShapeUtils::ShapeString(m_operatingGeometry);
		Utils::Logger::LogInfo(string);
	}
	Utils::Logger::LogTiming("Kernel analysis", timeKernel);
}

bool KernelAnalysis::VisitIn(const HorseIR::Statement *statement)
{
	auto statementGeometry = m_geometryAnalysis.GetGeometry(statement);
	m_operatingGeometry = GeometryUtils::MaxGeometry(m_operatingGeometry, statementGeometry);
	return false;
}

}
