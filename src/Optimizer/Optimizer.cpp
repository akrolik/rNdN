#include "Optimizer/Optimizer.h"

#include "Analysis/BasicFlow/ReachingDefinitions.h"
#include "Analysis/BasicFlow/LiveVariables.h"
#include "Analysis/BasicFlow/UDDUChainsBuilder.h"

#include "Utils/Chrono.h"

namespace Optimizer {

void Optimizer::Optimize(HorseIR::Program *program)
{
	auto timeOptimizer_start = Utils::Chrono::Start("Optimizer");
	m_program = program;
	program->Accept(*this);
	Utils::Chrono::End(timeOptimizer_start);
}

bool Optimizer::VisitIn(HorseIR::Function *function)
{
	auto timeOptimizer_start = Utils::Chrono::Start("Optimizer '" + function->GetName() + "'");

	// Reaching definitions

	Analysis::ReachingDefinitions reachingDefs(m_program);
	reachingDefs.Analyze(function);

	// UD/DU chain builder
	
	Analysis::UDDUChainsBuilder useDefs(reachingDefs);
	useDefs.Build(function);

	// Live variables

	Analysis::LiveVariables liveVariables(m_program);
	liveVariables.Analyze(function);

	Utils::Chrono::End(timeOptimizer_start);

	return false;
}

}
