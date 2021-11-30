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

template<class ISETPInstruction, class T>
typename ISETPInstruction::ComparisonOperator SetPredicateGenerator::IInstructionComparisonOperator(const PTX::SetPredicateInstruction<T> *instruction)
{
	if constexpr(PTX::is_bit_type<T>::value)
	{
		switch (instruction->GetComparisonOperator())
		{
			case T::ComparisonOperator::Equal:
			{
				return ISETPInstruction::ComparisonOperator::EQ;
			}
			case T::ComparisonOperator::NotEqual:
			{
				return ISETPInstruction::ComparisonOperator::NE;
			}
		}
	}
	else if constexpr(PTX::is_unsigned_int_type<T>::value)
	{
		switch (instruction->GetComparisonOperator())
		{
			case T::ComparisonOperator::Equal:
			{
				return ISETPInstruction::ComparisonOperator::EQ;
			}
			case T::ComparisonOperator::NotEqual:
			{
				return ISETPInstruction::ComparisonOperator::NE;
			}
			case T::ComparisonOperator::Less:
			case T::ComparisonOperator::Lower:
			{
				return ISETPInstruction::ComparisonOperator::LT;
			}
			case T::ComparisonOperator::LessEqual:
			case T::ComparisonOperator::LowerSame:
			{
				return ISETPInstruction::ComparisonOperator::LE;
			}
			case T::ComparisonOperator::Greater:
			case T::ComparisonOperator::Higher:
			{
				return ISETPInstruction::ComparisonOperator::GT;
			}
			case T::ComparisonOperator::GreaterEqual:
			case T::ComparisonOperator::HigherSame:
			{
				return ISETPInstruction::ComparisonOperator::GE;
			}
		}
	}
	else if constexpr(PTX::is_signed_int_type<T>::value)
	{
		switch (instruction->GetComparisonOperator())
		{
			case T::ComparisonOperator::Equal:
			{
				return ISETPInstruction::ComparisonOperator::EQ;
			}
			case T::ComparisonOperator::NotEqual:
			{
				return ISETPInstruction::ComparisonOperator::NE;
			}
			case T::ComparisonOperator::Less:
			{
				return ISETPInstruction::ComparisonOperator::LT;
			}
			case T::ComparisonOperator::LessEqual:
			{
				return ISETPInstruction::ComparisonOperator::LE;
			}
			case T::ComparisonOperator::Greater:
			{
				return ISETPInstruction::ComparisonOperator::GT;
			}
			case T::ComparisonOperator::GreaterEqual:
			{
				return ISETPInstruction::ComparisonOperator::GE;
			}
		}
	}
	Error(instruction, "unsupported comparison operation");
}

template<class DSETPInstruction, class T>
typename DSETPInstruction::ComparisonOperator SetPredicateGenerator::DInstructionComparisonOperator(const PTX::SetPredicateInstruction<T> *instruction)
{
	if constexpr(PTX::is_float_type<T>::value && T::TypeBits == PTX::Bits::Bits64)
	{
		switch (instruction->GetComparisonOperator())
		{
			// Ordered
			case T::ComparisonOperator::Equal:
			{
				return DSETPInstruction::ComparisonOperator::EQ;
			}
			case T::ComparisonOperator::NotEqual:
			{
				return DSETPInstruction::ComparisonOperator::NE;
			}
			case T::ComparisonOperator::Less:
			{
				return DSETPInstruction::ComparisonOperator::LT;
			}
			case T::ComparisonOperator::LessEqual:
			{
				return DSETPInstruction::ComparisonOperator::LE;
			}
			case T::ComparisonOperator::Greater:
			{
				return DSETPInstruction::ComparisonOperator::GT;
			}
			case T::ComparisonOperator::GreaterEqual:
			{
				return DSETPInstruction::ComparisonOperator::GE;
			}

			// Unordered
			case T::ComparisonOperator::EqualUnordered:
			{
				return DSETPInstruction::ComparisonOperator::EQU;
			}
			case T::ComparisonOperator::NotEqualUnordered:
			{
				return DSETPInstruction::ComparisonOperator::NEU;
			}
			case T::ComparisonOperator::LessUnordered:
			{
				return DSETPInstruction::ComparisonOperator::LTU;
			}
			case T::ComparisonOperator::LessEqualUnordered:
			{
				return DSETPInstruction::ComparisonOperator::LEU;
			}
			case T::ComparisonOperator::GreaterUnordered:
			{
				return DSETPInstruction::ComparisonOperator::GTU;
			}
			case T::ComparisonOperator::GreaterEqualUnordered:
			{
				return DSETPInstruction::ComparisonOperator::GEU;
			}

			// Special
			case T::ComparisonOperator::Number:
			{
				return DSETPInstruction::ComparisonOperator::NUM;
			}
			case T::ComparisonOperator::NaN:
			{
				return DSETPInstruction::ComparisonOperator::NaN;
			}
		}
	}
	Error(instruction, "unsupported comparison operation");
}

template<class SETPInstruction, class T>
typename SETPInstruction::BooleanOperator SetPredicateGenerator::InstructionBooleanOperator(const PTX::SetPredicateInstruction<T> *instruction)
{
	// Boolean operation (for source C)

	switch (instruction->GetBoolOperator())
	{
		case PTX::SetPredicateInstruction<T>::BoolOperator::And:
		{
			return SETPInstruction::BooleanOperator::AND;
		}
		case PTX::SetPredicateInstruction<T>::BoolOperator::Or:
		{
			return SETPInstruction::BooleanOperator::XOR;
		}
		case PTX::SetPredicateInstruction<T>::BoolOperator::Xor:
		{
			return SETPInstruction::BooleanOperator::XOR;
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
	GenerateInstruction<SASS::Maxwell::ISETPInstruction, SASS::Maxwell::DSETPInstruction, SASS::Maxwell::PRMTInstruction>(instruction);
}

template<class T>
void SetPredicateGenerator::GenerateVolta(const PTX::SetPredicateInstruction<T> *instruction)
{
	GenerateInstruction<SASS::Volta::ISETPInstruction, SASS::Volta::DSETPInstruction, SASS::Volta::PRMTInstruction>(instruction);
}

template<class ISETPInstruction, class DSETPInstruction, class PRMTInstruction, class T>
void SetPredicateGenerator::GenerateInstruction(const PTX::SetPredicateInstruction<T> *instruction)
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

	RegisterGenerator registerGenerator(this->m_builder);
	CompositeGenerator compositeGenerator(this->m_builder);

	if constexpr(std::is_same<ISETPInstruction, SASS::Volta::ISETPInstruction>::value)
	{
		compositeGenerator.SetImmediateSize(32);
	}

	// Generate instruction

	if constexpr(PTX::is_int_type<T>::value || PTX::is_bit_type<T>::value)
	{
		// Source operands

		auto [sourceA_Lo, sourceA_Hi] = registerGenerator.GeneratePair(instruction->GetSourceA());
		auto [sourceB_Lo, sourceB_Hi] = compositeGenerator.GeneratePair(instruction->GetSourceB());

		// Comparison/boolean operators

		auto comparisonOperator = IInstructionComparisonOperator<ISETPInstruction, T>(instruction);
		auto booleanOperator = InstructionBooleanOperator<ISETPInstruction, T>(instruction);

		// Flags

		auto flags = ISETPInstruction::Flags::None;
		if (instruction->GetNegateSourcePredicate() ^ sourceC_Not)
		{
			flags |= ISETPInstruction::Flags::NOT_C;
		}

		// All unsigned ints get flag (as the register size is always 32-bit)

		if constexpr(PTX::is_unsigned_int_type<T>::value || PTX::is_bit_type<T>::value)
		{
			flags |= ISETPInstruction::Flags::U32;
		}

		// Conversions required for smaller/larger integers

		if constexpr(std::is_same<T, PTX::Int16Type>::value)
		{
			// Temporary variables for conversions to 32-bit

			auto temp0 = this->m_builder.AllocateTemporaryRegister();
			auto temp1 = this->m_builder.AllocateTemporaryRegister();

			auto [sourceB_Lo, sourceB_Hi] = registerGenerator.GeneratePair(instruction->GetSourceB());

			this->AddInstruction(new PRMTInstruction(temp0, sourceA_Lo, new SASS::I32Immediate(0x9910), SASS::RZ));
			this->AddInstruction(new PRMTInstruction(temp1, sourceB_Lo, new SASS::I32Immediate(0x9910), SASS::RZ));

			sourceA_Lo = temp0;
			sourceB_Lo = temp1;
		}
		else if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			if constexpr(std::is_same<ISETPInstruction, SASS::Volta::ISETPInstruction>::value)
			{
				// Dummy operation for carry bit, used in ISETP.EX instruction. Always unsigned

				auto temp = this->m_builder.AllocateTemporaryPredicate();

				this->AddInstruction(new ISETPInstruction(
					temp, SASS::PT, sourceA_Lo, sourceB_Lo, SASS::PT,
					comparisonOperator, ISETPInstruction::BooleanOperator::AND, ISETPInstruction::Flags::U32
				));

				this->AddInstruction(new ISETPInstruction(
					destinationA, destinationB, sourceA_Hi, sourceB_Hi, sourceC, temp,
					comparisonOperator, booleanOperator, flags | ISETPInstruction::Flags::EX
				));
			}
			else
			{
				// Dummy operation for carry bit, used in the ISETP instruction (.X flag)
				//   IADD RZ.CC, R0, -R1

				this->AddInstruction(new SASS::Maxwell::IADDInstruction(
					SASS::RZ, sourceA_Lo, sourceB_Lo,
					SASS::Maxwell::IADDInstruction::Flags::NEG_B | SASS::Maxwell::IADDInstruction::Flags::CC
				));

				flags |= ISETPInstruction::Flags::X;

				// Finally, the instruction

				this->AddInstruction(new ISETPInstruction(
					destinationA, destinationB, sourceA_Hi, sourceB_Hi, sourceC, comparisonOperator, booleanOperator, flags
				));
			}
			return;
		}
		
		// Finally, the instruction

		this->AddInstruction(new ISETPInstruction(
			destinationA, destinationB, sourceA_Lo, sourceB_Lo, sourceC, comparisonOperator, booleanOperator, flags
		));
	}
	else if constexpr(std::is_same<T, PTX::Float64Type>::value)
	{
		// Source operands

		auto sourceA = registerGenerator.Generate(instruction->GetSourceA());
		auto sourceB = compositeGenerator.Generate(instruction->GetSourceB());

		// Comparison/boolean operators

		auto comparisonOperator = DInstructionComparisonOperator<DSETPInstruction, T>(instruction);
		auto booleanOperator = InstructionBooleanOperator<DSETPInstruction, T>(instruction);

		// Flags

		auto flags = DSETPInstruction::Flags::None;
		if (instruction->GetNegateSourcePredicate() ^ sourceC_Not)
		{
			flags |= DSETPInstruction::Flags::NOT_C;
		}

		// Generate instruction

		this->AddInstruction(new DSETPInstruction(
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
