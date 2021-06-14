#include "SASS/Optimizer/Optimizer.h"

#include "SASS/Transformation/RedundantCodeElimination.h"
#include "SASS/Transformation/DeadLoadElimination.h"

#include "Utils/Chrono.h"

namespace SASS {
namespace Optimizer {

void Optimizer::Optimize(Function *function)
{
	auto timeOptimizer_start = Utils::Chrono::Start("Backend optimizer '" + function->GetName() + "'");

	// Redundant code elimination (MOV R2, R2)

	Transformation::RedundantCodeElimination redundantCode;
	redundantCode.Transform(function);

	// Dead load elimination (LDG RZ, [RX])

	Transformation::DeadLoadElimination deadLoad;
	deadLoad.Transform(function);

	Utils::Chrono::End(timeOptimizer_start);
}

}
}
