#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"

#include "PTX/Utils/PrettyPrinter.h"

namespace Backend {
namespace Codegen {

SASS::Predicate *PredicateGenerator::Generate(const PTX::Operand *operand)
{
	// Clear

	m_predicate = nullptr;

	// Generate predicate

	operand->Accept(*this);
	if (m_predicate == nullptr)
	{
		Error("predicate for operand '" + PTX::PrettyPrinter::PrettyString(operand) + "'");
	}
	return m_predicate;
}


void PredicateGenerator::Visit(const PTX::_Register *reg)
{
	reg->Dispatch(*this);
}

template<class T>
void PredicateGenerator::Visit(const PTX::Register<T> *reg)
{
	if constexpr(std::is_same<T, PTX::PredicateType>::value)
	{
		const auto& allocations = this->m_builder.GetRegisterAllocation();

		// Verify predicate allocated

		const auto& name = reg->GetName();
		if (allocations->ContainsPredicate(name))
		{
			// Create SASS predicate

			const auto& allocation = allocations->GetPredicate(name);
			m_predicate = new SASS::Predicate(allocation);
		}
	}
}

void PredicateGenerator::Visit(const PTX::_Value *value)
{
	value->Dispatch(*this);
}

template<class T>
void PredicateGenerator::Visit(const PTX::Value<T> *value)
{
	if (value->GetValue() == true)
	{
		m_predicate = SASS::PT;
	}
	//TODO: Other values
}

}
}
