#include "Backend/Codegen/Generators/Instructions/Comparison/SetPredicateGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"
#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void SetPredicateGenerator::Generate(const PTX::_SetPredicateInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class I, class T>
typename I::ComparisonOperator SetPredicateGenerator::IInstructionComparisonOperator(const PTX::SetPredicateInstruction<T> *instruction)
{
	if constexpr(PTX::is_bit_type<T>::value)
	{
		switch (instruction->GetComparisonOperator())
		{
			case T::ComparisonOperator::Equal:
			{
				return I::ComparisonOperator::EQ;
			}
			case T::ComparisonOperator::NotEqual:
			{
				return I::ComparisonOperator::NE;
			}
		}
	}
	else if constexpr(PTX::is_unsigned_int_type<T>::value)
	{
		switch (instruction->GetComparisonOperator())
		{
			case T::ComparisonOperator::Equal:
			{
				return I::ComparisonOperator::EQ;
			}
			case T::ComparisonOperator::NotEqual:
			{
				return I::ComparisonOperator::NE;
			}
			case T::ComparisonOperator::Less:
			case T::ComparisonOperator::Lower:
			{
				return I::ComparisonOperator::LT;
			}
			case T::ComparisonOperator::LessEqual:
			case T::ComparisonOperator::LowerSame:
			{
				return I::ComparisonOperator::LE;
			}
			case T::ComparisonOperator::Greater:
			case T::ComparisonOperator::Higher:
			{
				return I::ComparisonOperator::GT;
			}
			case T::ComparisonOperator::GreaterEqual:
			case T::ComparisonOperator::HigherSame:
			{
				return I::ComparisonOperator::GE;
			}
		}
	}
	else if constexpr(PTX::is_signed_int_type<T>::value)
	{
		switch (instruction->GetComparisonOperator())
		{
			case T::ComparisonOperator::Equal:
			{
				return I::ComparisonOperator::EQ;
			}
			case T::ComparisonOperator::NotEqual:
			{
				return I::ComparisonOperator::NE;
			}
			case T::ComparisonOperator::Less:
			{
				return I::ComparisonOperator::LT;
			}
			case T::ComparisonOperator::LessEqual:
			{
				return I::ComparisonOperator::LE;
			}
			case T::ComparisonOperator::Greater:
			{
				return I::ComparisonOperator::GT;
			}
			case T::ComparisonOperator::GreaterEqual:
			{
				return I::ComparisonOperator::GE;
			}
		}
	}
	Error(instruction, "unsupported comparison operation");
}

template<class I, class T>
typename I::ComparisonOperator SetPredicateGenerator::DInstructionComparisonOperator(const PTX::SetPredicateInstruction<T> *instruction)
{
	if constexpr(PTX::is_float_type<T>::value && T::TypeBits == PTX::Bits::Bits64)
	{
		switch (instruction->GetComparisonOperator())
		{
			// Ordered
			case T::ComparisonOperator::Equal:
			{
				return I::ComparisonOperator::EQ;
			}
			case T::ComparisonOperator::NotEqual:
			{
				return I::ComparisonOperator::NE;
			}
			case T::ComparisonOperator::Less:
			{
				return I::ComparisonOperator::LT;
			}
			case T::ComparisonOperator::LessEqual:
			{
				return I::ComparisonOperator::LE;
			}
			case T::ComparisonOperator::Greater:
			{
				return I::ComparisonOperator::GT;
			}
			case T::ComparisonOperator::GreaterEqual:
			{
				return I::ComparisonOperator::GE;
			}

			// Unordered
			case T::ComparisonOperator::EqualUnordered:
			{
				return I::ComparisonOperator::EQU;
			}
			case T::ComparisonOperator::NotEqualUnordered:
			{
				return I::ComparisonOperator::NEU;
			}
			case T::ComparisonOperator::LessUnordered:
			{
				return I::ComparisonOperator::LTU;
			}
			case T::ComparisonOperator::LessEqualUnordered:
			{
				return I::ComparisonOperator::LEU;
			}
			case T::ComparisonOperator::GreaterUnordered:
			{
				return I::ComparisonOperator::GTU;
			}
			case T::ComparisonOperator::GreaterEqualUnordered:
			{
				return I::ComparisonOperator::GEU;
			}

			// Special
			case T::ComparisonOperator::Number:
			{
				return I::ComparisonOperator::NUM;
			}
			case T::ComparisonOperator::NaN:
			{
				return I::ComparisonOperator::NaN;
			}
		}
	}
	Error(instruction, "unsupported comparison operation");
}

template<class I, class T>
typename I::BooleanOperator SetPredicateGenerator::IInstructionBooleanOperator(const PTX::SetPredicateInstruction<T> *instruction)
{
	// Boolean operation (for source C)

	switch (instruction->GetBoolOperator())
	{
		case PTX::SetPredicateInstruction<T>::BoolOperator::And:
		{
			return I::BooleanOperator::AND;
		}
		case PTX::SetPredicateInstruction<T>::BoolOperator::Or:
		{
			return I::BooleanOperator::XOR;
		}
		case PTX::SetPredicateInstruction<T>::BoolOperator::Xor:
		{
			return I::BooleanOperator::XOR;
		}
	}
	Error(instruction, "unsupported boolean operation");
}
	
template<class I, class T>
typename I::BooleanOperator SetPredicateGenerator::DInstructionBooleanOperator(const PTX::SetPredicateInstruction<T> *instruction)
{
	// Boolean operation (for source C)

	switch (instruction->GetBoolOperator())
	{
		case PTX::SetPredicateInstruction<T>::BoolOperator::And:
		{
			return I::BooleanOperator::AND;
		}
		case PTX::SetPredicateInstruction<T>::BoolOperator::Or:
		{
			return I::BooleanOperator::XOR;
		}
		case PTX::SetPredicateInstruction<T>::BoolOperator::Xor:
		{
			return I::BooleanOperator::XOR;
		}
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

	ArchitectureDispatch::Dispatch(*this, instruction);
}

template<class T>
void SetPredicateGenerator::GenerateMaxwell(const PTX::SetPredicateInstruction<T> *instruction)
{
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

		auto comparisonOperator = IInstructionComparisonOperator<SASS::Maxwell::ISETPInstruction, T>(instruction);
		auto booleanOperator = IInstructionBooleanOperator<SASS::Maxwell::ISETPInstruction, T>(instruction);

		// Flags

		auto flags = SASS::Maxwell::ISETPInstruction::Flags::None;
		if (instruction->GetNegateSourcePredicate() ^ sourceC_Not)
		{
			flags |= SASS::Maxwell::ISETPInstruction::Flags::NOT_C;
		}

		// All unsigned ints get flag (as the register size is always 32-bit)

		if constexpr(PTX::is_unsigned_int_type<T>::value || PTX::is_bit_type<T>::value)
		{
			flags |= SASS::Maxwell::ISETPInstruction::Flags::U32;
		}

		// Conversions required for smaller/larger integers

		if constexpr(T::TypeBits == PTX::Bits::Bits16)
		{
			// Temporary variables for conversions to 32-bit

			auto temp0 = this->m_builder.AllocateTemporaryRegister();
			auto temp1 = this->m_builder.AllocateTemporaryRegister();

			if constexpr(PTX::is_unsigned_int_type<T>::value || PTX::is_bit_type<T>::value)
			{
				this->AddInstruction(new SASS::Maxwell::MOVInstruction(temp0, sourceA_Lo));
				this->AddInstruction(new SASS::Maxwell::MOVInstruction(temp1, sourceB_Lo));
			}
			else
			{
				auto [sourceB_Lo, sourceB_Hi] = registerGenerator.GeneratePair(instruction->GetSourceB());

				this->AddInstruction(new SASS::Maxwell::BFEInstruction(temp0, sourceA_Lo, new SASS::I32Immediate(0x1000)));
				this->AddInstruction(new SASS::Maxwell::BFEInstruction(temp1, sourceB_Lo, new SASS::I32Immediate(0x1000)));
			}

			sourceA_Lo = temp0;
			sourceB_Lo = temp1;
		}
		else if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			// Dummy operation for carry bit, used in the ISETP instruction (.X flag)
			//   IADD RZ.CC, R0, -R1

			this->AddInstruction(new SASS::Maxwell::IADDInstruction(
				SASS::RZ, sourceA_Lo, sourceB_Lo,
				SASS::Maxwell::IADDInstruction::Flags::NEG_B | SASS::Maxwell::IADDInstruction::Flags::CC
			));

			flags |= SASS::Maxwell::ISETPInstruction::Flags::X;

			// Finally, the instruction

			this->AddInstruction(new SASS::Maxwell::ISETPInstruction(
				destinationA, destinationB, sourceA_Hi, sourceB_Hi, sourceC, comparisonOperator, booleanOperator, flags
			));

			return;
		}
		
		// Finally, the instruction

		this->AddInstruction(new SASS::Maxwell::ISETPInstruction(
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

		auto comparisonOperator = DInstructionComparisonOperator<SASS::Maxwell::DSETPInstruction, T>(instruction);
		auto booleanOperator = DInstructionBooleanOperator<SASS::Maxwell::DSETPInstruction, T>(instruction);

		// Flags

		auto flags = SASS::Maxwell::DSETPInstruction::Flags::None;
		if (instruction->GetNegateSourcePredicate())
		{
			flags |= SASS::Maxwell::DSETPInstruction::Flags::NOT_C;
		}

		// Generate instruction

		this->AddInstruction(new SASS::Maxwell::DSETPInstruction(
			destinationA, destinationB, sourceA, sourceB, sourceC, comparisonOperator, booleanOperator, flags
		));
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

template<class T>
void SetPredicateGenerator::GenerateVolta(const PTX::SetPredicateInstruction<T> *instruction)
{
	Error(instruction, "unsupported architecture");
}

}
}
