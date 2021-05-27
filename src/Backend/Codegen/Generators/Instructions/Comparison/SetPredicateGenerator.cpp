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
SASS::ISETPInstruction::ComparisonOperator SetPredicateGenerator::IInstructionComparisonOperator(const PTX::SetPredicateInstruction<T> *instruction)
{
	if constexpr(PTX::is_bit_type<T>::value)
	{
		switch (instruction->GetComparisonOperator())
		{
			case T::ComparisonOperator::Equal:
			{
				return SASS::ISETPInstruction::ComparisonOperator::EQ;
			}
			case T::ComparisonOperator::NotEqual:
			{
				return SASS::ISETPInstruction::ComparisonOperator::NE;
			}
		}
	}
	else if constexpr(PTX::is_unsigned_int_type<T>::value)
	{
		switch (instruction->GetComparisonOperator())
		{
			case T::ComparisonOperator::Equal:
			{
				return SASS::ISETPInstruction::ComparisonOperator::EQ;
			}
			case T::ComparisonOperator::NotEqual:
			{
				return SASS::ISETPInstruction::ComparisonOperator::NE;
			}
			case T::ComparisonOperator::Less:
			case T::ComparisonOperator::Lower:
			{
				return SASS::ISETPInstruction::ComparisonOperator::LT;
			}
			case T::ComparisonOperator::LessEqual:
			case T::ComparisonOperator::LowerSame:
			{
				return SASS::ISETPInstruction::ComparisonOperator::LE;
			}
			case T::ComparisonOperator::Greater:
			case T::ComparisonOperator::Higher:
			{
				return SASS::ISETPInstruction::ComparisonOperator::GT;
			}
			case T::ComparisonOperator::GreaterEqual:
			case T::ComparisonOperator::HigherSame:
			{
				return SASS::ISETPInstruction::ComparisonOperator::GE;
			}
		}
	}
	else if constexpr(PTX::is_signed_int_type<T>::value)
	{
		switch (instruction->GetComparisonOperator())
		{
			case T::ComparisonOperator::Equal:
			{
				return SASS::ISETPInstruction::ComparisonOperator::EQ;
			}
			case T::ComparisonOperator::NotEqual:
			{
				return SASS::ISETPInstruction::ComparisonOperator::NE;
			}
			case T::ComparisonOperator::Less:
			{
				return SASS::ISETPInstruction::ComparisonOperator::LT;
			}
			case T::ComparisonOperator::LessEqual:
			{
				return SASS::ISETPInstruction::ComparisonOperator::LE;
			}
			case T::ComparisonOperator::Greater:
			{
				return SASS::ISETPInstruction::ComparisonOperator::GT;
			}
			case T::ComparisonOperator::GreaterEqual:
			{
				return SASS::ISETPInstruction::ComparisonOperator::GE;
			}
		}
	}
	Error(instruction, "unsupported comparison operation");
}

template<class T>
SASS::DSETPInstruction::ComparisonOperator SetPredicateGenerator::DInstructionComparisonOperator(const PTX::SetPredicateInstruction<T> *instruction)
{
	if constexpr(PTX::is_float_type<T>::value && T::TypeBits == PTX::Bits::Bits64)
	{
		switch (instruction->GetComparisonOperator())
		{
			// Ordered
			case T::ComparisonOperator::Equal:
			{
				return SASS::DSETPInstruction::ComparisonOperator::EQ;
			}
			case T::ComparisonOperator::NotEqual:
			{
				return SASS::DSETPInstruction::ComparisonOperator::NE;
			}
			case T::ComparisonOperator::Less:
			{
				return SASS::DSETPInstruction::ComparisonOperator::LT;
			}
			case T::ComparisonOperator::LessEqual:
			{
				return SASS::DSETPInstruction::ComparisonOperator::LE;
			}
			case T::ComparisonOperator::Greater:
			{
				return SASS::DSETPInstruction::ComparisonOperator::GT;
			}
			case T::ComparisonOperator::GreaterEqual:
			{
				return SASS::DSETPInstruction::ComparisonOperator::GE;
			}

			// Unordered
			case T::ComparisonOperator::EqualUnordered:
			{
				return SASS::DSETPInstruction::ComparisonOperator::EQU;
			}
			case T::ComparisonOperator::NotEqualUnordered:
			{
				return SASS::DSETPInstruction::ComparisonOperator::NEU;
			}
			case T::ComparisonOperator::LessUnordered:
			{
				return SASS::DSETPInstruction::ComparisonOperator::LTU;
			}
			case T::ComparisonOperator::LessEqualUnordered:
			{
				return SASS::DSETPInstruction::ComparisonOperator::LEU;
			}
			case T::ComparisonOperator::GreaterUnordered:
			{
				return SASS::DSETPInstruction::ComparisonOperator::GTU;
			}
			case T::ComparisonOperator::GreaterEqualUnordered:
			{
				return SASS::DSETPInstruction::ComparisonOperator::GEU;
			}

			// Special
			case T::ComparisonOperator::Number:
			{
				return SASS::DSETPInstruction::ComparisonOperator::NUM;
			}
			case T::ComparisonOperator::NaN:
			{
				return SASS::DSETPInstruction::ComparisonOperator::NaN;
			}
		}
	}
	Error(instruction, "unsupported comparison operation");
}

template<class T>
SASS::ISETPInstruction::BooleanOperator SetPredicateGenerator::IInstructionBooleanOperator(const PTX::SetPredicateInstruction<T> *instruction)
{
	// Boolean operation (for source C)

	switch (instruction->GetBoolOperator())
	{
		case PTX::SetPredicateInstruction<T>::BoolOperator::And:
			return SASS::ISETPInstruction::BooleanOperator::AND;
		case PTX::SetPredicateInstruction<T>::BoolOperator::Or:
			return SASS::ISETPInstruction::BooleanOperator::XOR;
		case PTX::SetPredicateInstruction<T>::BoolOperator::Xor:
			return SASS::ISETPInstruction::BooleanOperator::XOR;
	}
	Error(instruction, "unsupported boolean operation");
}
	
template<class T>
SASS::DSETPInstruction::BooleanOperator SetPredicateGenerator::DInstructionBooleanOperator(const PTX::SetPredicateInstruction<T> *instruction)
{
	// Boolean operation (for source C)

	switch (instruction->GetBoolOperator())
	{
		case PTX::SetPredicateInstruction<T>::BoolOperator::And:
			return SASS::DSETPInstruction::BooleanOperator::AND;
		case PTX::SetPredicateInstruction<T>::BoolOperator::Or:
			return SASS::DSETPInstruction::BooleanOperator::XOR;
		case PTX::SetPredicateInstruction<T>::BoolOperator::Xor:
			return SASS::DSETPInstruction::BooleanOperator::XOR;
	}
	Error(instruction, "unsupported boolean operation");
}
	
template<class T>
void SetPredicateGenerator::Visit(const PTX::SetPredicateInstruction<T> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types:
	//   - Bit16, Bit32, Bit64
	//   - Int16, Int32, Int64
	//   - UInt16, UInt32, UInt64
	//   - Float16, Float16x2, Float32, Float64
	// Modifiers:
	//   - Comparison: *
	//   - FlushSubnormal: Float16, Float16x2, Float32
	//   - Predicate: *

	// Generate operands

	PredicateGenerator predicateGenerator(this->m_builder);
	auto destinationA = predicateGenerator.Generate(instruction->GetDestination()).first;

	// Optional destination Q predicate

	auto destinationB_opt = instruction->GetDestinationQ();
	auto destinationB = (destinationB_opt == nullptr) ? SASS::PT : predicateGenerator.Generate(destinationB_opt).first;

	// Optional source C predicate

	auto sourceC_opt = instruction->GetSourcePredicate();
	auto [sourceC, sourceC_Not] = (sourceC_opt == nullptr) ? std::make_pair(SASS::PT, false) : predicateGenerator.Generate(sourceC_opt);

	// Generate instruction

	if constexpr(PTX::is_int_type<T>::value || PTX::is_bit_type<T>::value)
	{
		// Source operands

		RegisterGenerator registerGenerator(this->m_builder);
		auto [sourceA_Lo, sourceA_Hi] = registerGenerator.GeneratePair(instruction->GetSourceA());

		CompositeGenerator compositeGenerator(this->m_builder);
		auto [sourceB_Lo, sourceB_Hi] = compositeGenerator.GeneratePair(instruction->GetSourceB());

		// Comparison/boolean operators

		auto comparisonOperator = IInstructionComparisonOperator<T>(instruction);
		auto booleanOperator = IInstructionBooleanOperator<T>(instruction);

		// Flags

		auto flags = SASS::ISETPInstruction::Flags::None;
		if (instruction->GetNegateSourcePredicate() ^ sourceC_Not)
		{
			flags |= SASS::ISETPInstruction::Flags::NOT_C;
		}

		// All unsigned ints get flag (as the register size is always 32-bit)

		if constexpr(PTX::is_unsigned_int_type<T>::value || PTX::is_bit_type<T>::value)
		{
			flags |= SASS::ISETPInstruction::Flags::U32;
		}

		// Conversions required for smaller/larger integers

		if constexpr(T::TypeBits == PTX::Bits::Bits16)
		{
			// Temporary variables for conversions to 32-bit

			auto temp0 = this->m_builder.AllocateTemporaryRegister();
			auto temp1 = this->m_builder.AllocateTemporaryRegister();

			if constexpr(PTX::is_unsigned_int_type<T>::value || PTX::is_bit_type<T>::value)
			{
				this->AddInstruction(new SASS::I2IInstruction(
					temp0, sourceA_Lo, SASS::I2IInstruction::DestinationType::U32, SASS::I2IInstruction::SourceType::U16
				));
				this->AddInstruction(new SASS::I2IInstruction(
					temp1, sourceB_Lo, SASS::I2IInstruction::DestinationType::U32, SASS::I2IInstruction::SourceType::U16
				));
			}
			else
			{
				this->AddInstruction(new SASS::I2IInstruction(
					temp0, sourceA_Lo, SASS::I2IInstruction::DestinationType::S32, SASS::I2IInstruction::SourceType::S16
				));
				this->AddInstruction(new SASS::I2IInstruction(
					temp1, sourceB_Lo, SASS::I2IInstruction::DestinationType::S32, SASS::I2IInstruction::SourceType::S16
				));
			}

			sourceA_Lo = temp0;
			sourceB_Lo = temp1;
		}
		else if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			// Dummy operation for carry bit, used in the ISETP instruction (.X flag)
			//   IADD RZ.CC, R0, -R1

			this->AddInstruction(new SASS::IADDInstruction(
				SASS::RZ, sourceA_Lo, sourceB_Lo, SASS::IADDInstruction::Flags::NEG_B | SASS::IADDInstruction::Flags::CC
			));

			flags |= SASS::ISETPInstruction::Flags::X;

			// Finally, the instruction

			this->AddInstruction(new SASS::ISETPInstruction(
				destinationA, destinationB, sourceA_Hi, sourceB_Hi, sourceC, comparisonOperator, booleanOperator, flags
			));

			return;
		}
		
		// Finally, the instruction

		this->AddInstruction(new SASS::ISETPInstruction(
			destinationA, destinationB, sourceA_Lo, sourceB_Lo, sourceC, comparisonOperator, booleanOperator, flags
		));
	}
	else if constexpr(std::is_same<T, PTX::Float64Type>::value)
	{
		// Source operands

		RegisterGenerator registerGenerator(this->m_builder);
		auto sourceA = registerGenerator.Generate(instruction->GetSourceA());

		CompositeGenerator compositeGenerator(this->m_builder);
		auto sourceB = compositeGenerator.Generate(instruction->GetSourceB());

		// Comparison/boolean operators

		auto comparisonOperator = DInstructionComparisonOperator<T>(instruction);
		auto booleanOperator = DInstructionBooleanOperator<T>(instruction);

		// Flags

		auto flags = SASS::DSETPInstruction::Flags::None;
		if (instruction->GetNegateSourcePredicate())
		{
			flags |= SASS::DSETPInstruction::Flags::NOT_C;
		}

		// Generate instruction

		this->AddInstruction(new SASS::DSETPInstruction(
			destinationA, destinationB, sourceA, sourceB, sourceC, comparisonOperator, booleanOperator, flags
		));
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

}
}
