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

bool Optimizer::VisitIn(Function *function)
{
	function->Accept(static_cast<FunctionVisitor&>(*this));
	return false;
}

void Optimizer::Visit(_FunctionDeclaration *function)
{
	Utils::Logger::LogError("PTX optimizer requires function definition");
}

void Optimizer::Visit(_FunctionDefinition *function)
{
	function->Dispatch(*this);
}

template<class T, class S>
void Optimizer::Visit(FunctionDefinition<T, S> *function)
{
	if constexpr(std::is_same<T, VoidType>::value && std::is_same<S, ParameterSpace>::value)
	{
		Optimize(function);
	}
	else
	{
		Utils::Logger::LogError("PTX optimizer requires VoidType function");
	}
}

void Optimizer::Optimize(FunctionDefinition<VoidType> *function)
{
	Analysis::ControlFlowBuilder cfgBuilder;
	auto cfg = cfgBuilder.Analyze(function);

	function->SetControlFlowGraph(cfg);
	function->SetBasicBlocks(cfg->GetNodes());
	function->InvalidateStatements();

	// Parameter constant propagation/dead code

	auto timeOptimizer_start = Utils::Chrono::Start("Frontend optimizer '" + function->GetName() + "'");

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
}

}
}
