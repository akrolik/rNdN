#include "Optimizer/Optimizer.h"

#include "Analysis/BasicFlow/ReachingDefinitions.h"
#include "Analysis/BasicFlow/LiveVariables.h"
#include "Analysis/BasicFlow/UDDUChainsBuilder.h"

#include "Utils/Logger.h"

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

	Analysis::ReachingDefinitions reachingDefs(m_program);
	reachingDefs.Analyze(function);

	// UD/DU chain builder
	
	Analysis::UDDUChainsBuilder useDefs(reachingDefs);
	useDefs.Build(function);

	// Live variables

	Analysis::LiveVariables liveVariables(m_program);
	liveVariables.Analyze(function);

	return false;
}

}
