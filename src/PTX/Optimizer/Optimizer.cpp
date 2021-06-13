#include "PTX/Optimizer/Optimizer.h"

#include "PTX/Analysis/BasicFlow/DefinitionsAnalysis.h"
#include "PTX/Analysis/BasicFlow/LiveVariables.h"
#include "PTX/Analysis/ControlFlow/ControlFlowBuilder.h"

#include "PTX/Transformation/ConstantPropagation/ParameterPropagation.h"
#include "PTX/Transformation/DeadCode/DeadCodeElimination.h"

#include "Utils/Chrono.h"

namespace PTX {
namespace Optimizer {

void Optimizer::Optimize(Program *program)
{
	program->Accept(*this);
}

bool Optimizer::VisitIn(FunctionDefinition<VoidType> *function)
{
	Analysis::ControlFlowBuilder cfgBuilder;
	auto cfg = cfgBuilder.Analyze(function);

	function->SetControlFlowGraph(cfg);
	function->SetBasicBlocks(cfg->GetNodes());
	function->InvalidateStatements();

	// Parameter constant propagation/dead code

	auto timeOptimizer_start = Utils::Chrono::Start("Optimizer '" + function->GetName() + "'");

	Analysis::DefinitionsAnalysis definitionsAnalysis;
	definitionsAnalysis.Analyze(function);

	Transformation::ParameterPropagation parameterPropagation(definitionsAnalysis);
	parameterPropagation.Transform(function);

	auto transform = true;
	while (transform)
	{
		Analysis::LiveVariables liveVariables;
		liveVariables.SetCollectInSets(false);
		liveVariables.Analyze(function);

		Transformation::DeadCodeElimination deadCode(liveVariables);
		transform = deadCode.Transform(function);
	}

	Utils::Chrono::End(timeOptimizer_start);

	return false;
}

}
}
