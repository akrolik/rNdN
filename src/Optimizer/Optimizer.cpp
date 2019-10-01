#include "Optimizer/Optimizer.h"

#include "Analysis/BasicFlow/ReachingDefinitions.h"
#include "Analysis/BasicFlow/LiveVariables.h"
#include "Analysis/BasicFlow/UDDUChainsBuilder.h"

#include "HorseIR/Analysis/FlowAnalysisPrinter.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Optimizer {

void Optimizer::Optimize(HorseIR::Program *program)
{
	m_program = program;
	program->Accept(*this);
}

bool Optimizer::VisitIn(HorseIR::Function *function)
{
	Utils::Logger::LogSection("Optimizing function '" + function->GetName() + "'");

	// Reaching definitions

	auto timeReachingDefs_start = Utils::Chrono::Start();

	Analysis::ReachingDefinitions reachingDefs(m_program);
	reachingDefs.Analyze(function);

	auto timeReachingDefs = Utils::Chrono::End(timeReachingDefs_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_analysis))
	{
		Utils::Logger::LogInfo("Reaching definitions");

		auto reachingDefsString = HorseIR::FlowAnalysisPrinter<Analysis::ReachingDefinitionsProperties>::PrettyString(reachingDefs, function);
		Utils::Logger::LogInfo(reachingDefsString, 0, true, Utils::Logger::NoPrefix);
	}

	Utils::Logger::LogTiming("Reaching definitions", timeReachingDefs);

	// UD/DU chain builder
	
	auto timeUDDU_start = Utils::Chrono::Start();

	Analysis::UDDUChainsBuilder useDefs(reachingDefs);
	useDefs.Build(function);

	auto timeUDDU = Utils::Chrono::End(timeUDDU_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_analysis))
	{
		Utils::Logger::LogInfo("UD/DU chains");

		auto useDefsString = useDefs.DebugString();
		Utils::Logger::LogInfo(useDefsString, 0, true, Utils::Logger::NoPrefix);
	}

	Utils::Logger::LogTiming("UD/DU chains", timeUDDU);

	// Live variables

	auto timeLiveVariables_start = Utils::Chrono::Start();

	Analysis::LiveVariables liveVariables(m_program);
	liveVariables.Analyze(function);

	auto timeLiveVariables = Utils::Chrono::End(timeLiveVariables_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_analysis))
	{
		Utils::Logger::LogInfo("Live variables");

		auto liveVariablesString = HorseIR::FlowAnalysisPrinter<Analysis::LiveVariablesProperties>::PrettyString(liveVariables, function);
		Utils::Logger::LogInfo(liveVariablesString, 0, true, Utils::Logger::NoPrefix);
	}

	Utils::Logger::LogTiming("Live variables", timeLiveVariables);

	return false;
}

}
