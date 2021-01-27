#include "Backend/Codegen/Generators/Instructions/Comparison/SetPredicateGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"
#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"

namespace Backend {
namespace Codegen {

void SetPredicateGenerator::Generate(const PTX::_SetPredicateInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void SetPredicateGenerator::Visit(const PTX::SetPredicateInstruction<T> *instruction)
{
	// Generate operands

	PredicateGenerator predicateGenerator(this->m_builder);
	auto destinationA = predicateGenerator.Generate(instruction->GetDestination());

	// Optional destination Q predicate
	auto destinationB_opt = instruction->GetDestinationQ();
	auto destinationB = (destinationB_opt == nullptr) ? SASS::PT : predicateGenerator.Generate(destinationB_opt);

	RegisterGenerator registerGenerator(this->m_builder);
	auto [sourceA, sourceA_Hi] = registerGenerator.Generate(instruction->GetSourceA());

	CompositeGenerator compositeGenerator(this->m_builder);
	auto [sourceB, sourceB_Hi] = compositeGenerator.Generate(instruction->GetSourceB());

	// Optional source C predicate
	auto sourceC_opt = instruction->GetSourcePredicate();
	auto sourceC = (sourceC_opt == nullptr) ? SASS::PT : predicateGenerator.Generate(sourceC_opt);

	// Comparison operator

	//TODO: Comparison operators for more types
	SASS::ISETPInstruction::ComparisonOperator comparisonOperator;
	if constexpr(std::is_same<T, PTX::UInt32Type>::value)
	{
		switch (instruction->GetComparisonOperator())
		{
			case T::ComparisonOperator::Equal:
			{
				comparisonOperator = SASS::ISETPInstruction::ComparisonOperator::EQ;
				break;
			}
			case T::ComparisonOperator::NotEqual:
			{
				comparisonOperator = SASS::ISETPInstruction::ComparisonOperator::NE;
				break;
			}
			case T::ComparisonOperator::Less:
			{
				comparisonOperator = SASS::ISETPInstruction::ComparisonOperator::LT;
				break;
			}
			case T::ComparisonOperator::LessEqual:
			{
				comparisonOperator = SASS::ISETPInstruction::ComparisonOperator::LE;
				break;
			}
			case T::ComparisonOperator::Greater:
			{
				comparisonOperator = SASS::ISETPInstruction::ComparisonOperator::GT;
				break;
			}
			case T::ComparisonOperator::GreaterEqual:
			{
				comparisonOperator = SASS::ISETPInstruction::ComparisonOperator::GE;
				break;
			}
		}
	}

	// Boolean operation (for source C)

	SASS::ISETPInstruction::BooleanOperator booleanOperator;
	switch (instruction->GetBoolOperator())
	{
		case PTX::SetPredicateInstruction<T>::BoolOperator::And:
		{
			booleanOperator = SASS::ISETPInstruction::BooleanOperator::AND;
			break;
		}
		case PTX::SetPredicateInstruction<T>::BoolOperator::Or:
		{
			booleanOperator = SASS::ISETPInstruction::BooleanOperator::XOR;
			break;
		}
		case PTX::SetPredicateInstruction<T>::BoolOperator::Xor:
		{
			booleanOperator = SASS::ISETPInstruction::BooleanOperator::XOR;
			break;
		}
	}
	
	// Flags

	auto flags = SASS::ISETPInstruction::Flags::None;
	if (instruction->GetNegateSourcePredicate())
	{
		flags |= SASS::ISETPInstruction::Flags::NOT_C;
	}
	if constexpr(std::is_same<T, PTX::UInt32Type>::value)
	{
		flags |= SASS::ISETPInstruction::Flags::U32;
	}

	// Generate instruction
	//  - Types:
	//  	- Bit16, Bit32, Bit64
	//  	- Int16, Int32, Int64
	//  	- UInt16, UInt32, UInt64
	//  	- Float16, Float16x2, Float32, Float64
	//  - Modifiers:
	//  	- Comparison: *
	//  	- FlushSubnormal: Float16, Float16x2, Float32
	//  	- Predicate: *

	this->AddInstruction(new SASS::ISETPInstruction(
		destinationA, destinationB, sourceA, sourceB, sourceC, comparisonOperator, booleanOperator, flags
	));
}

}
}
