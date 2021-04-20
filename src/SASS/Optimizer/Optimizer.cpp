#include "SASS/Optimizer/Optimizer.h"

#include "SASS/Transformation/RedundantCodeElimination.h"

#include "Utils/Chrono.h"

namespace SASS {
namespace Optimizer {

void Optimizer::Optimize(Function *function)
{
	auto timeOptimizer_start = Utils::Chrono::Start("Optimizer '" + function->GetName() + "'");

	// Redundant code elimination (MOV R2, R2)

	Transformation::RedundantCodeElimination redundantCode;
	redundantCode.Transform(function);

	Utils::Chrono::End(timeOptimizer_start);
}

}
}
