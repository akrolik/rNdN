#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"

namespace Backend {
namespace Codegen {

SASS::Predicate *PredicateGenerator::Generate(const PTX::Register<PTX::PredicateType> *reg)
{
	const auto& allocations = this->m_builder.GetRegisterAllocation();
	const auto& predicateAllocation = allocations->GetPredicate(reg->GetName());
	return new SASS::Predicate(predicateAllocation);
}

}
}
