#include "HorseIR/Optimizer/Optimizer.h"

#include "HorseIR/Analysis/BasicFlow/ReachingDefinitions.h"
#include "HorseIR/Analysis/BasicFlow/LiveVariables.h"
#include "HorseIR/Analysis/BasicFlow/UDDUChainsBuilder.h"

#include "Utils/Chrono.h"

namespace HorseIR {
namespace Optimizer {

void Optimizer::Optimize(Program *program)
{
	auto timeOptimizer_start = Utils::Chrono::Start("Optimizer");
	m_program = program;
	program->Accept(*this);
	Utils::Chrono::End(timeOptimizer_start);
}

bool Optimizer::VisitIn(Function *function)
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
}
