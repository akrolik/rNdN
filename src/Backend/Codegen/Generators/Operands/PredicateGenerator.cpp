#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"

#include "PTX/Utils/PrettyPrinter.h"

namespace Backend {
namespace Codegen {

std::pair<SASS::Predicate *, bool> PredicateGenerator::Generate(const PTX::Operand *operand)
{
	// Clear

	m_predicate = nullptr;
	m_negatePredicate = false;

	// Generate predicate

	operand->Accept(*this);
	if (m_predicate == nullptr)
	{
		Error(operand, "unsupported kind");
	}
	return { m_predicate, m_negatePredicate };
}


bool PredicateGenerator::Visit(const PTX::_Register *reg)
{
	reg->Dispatch(*this);
	return false;
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
			m_negatePredicate = false;
		}
		else
		{
			Error(reg, "predicate not found");
		}
	}
	else
	{
		Error(reg, "unsupported type");
	}
}

bool PredicateGenerator::Visit(const PTX::_Value *value)
{
	value->Dispatch(*this);
	return false;
}

template<class T>
void PredicateGenerator::Visit(const PTX::Value<T> *value)
{
	if constexpr(std::is_same<T, PTX::PredicateType>::value)
	{
		// Use the built-in PT register for true/false values

		m_predicate = SASS::PT;
		m_negatePredicate = (value->GetValue() == false);
	}
	else
	{
		Error(value, "unsupported type");
	}
}

}
}
