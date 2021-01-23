#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"

#include "PTX/Utils/PrettyPrinter.h"

namespace Backend {
namespace Codegen {

SASS::Predicate *PredicateGenerator::Generate(const PTX::Register<PTX::PredicateType> *predicate)
{
	const auto& allocations = this->m_builder.GetRegisterAllocation();

	// Verify predicate allocated

	const auto& name = predicate->GetName();
	if (!allocations->ContainsPredicate(name))
	{
		Error("predicate register for operand '" + PTX::PrettyPrinter::PrettyString(predicate) + "'");
	}

	// Create SASS predicate

	const auto& allocation = allocations->GetPredicate(name);
	return new SASS::Predicate(allocation);
}

}
}
