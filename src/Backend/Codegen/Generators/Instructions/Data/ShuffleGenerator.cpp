#include "Backend/Codegen/Generators/Instructions/Data/ShuffleGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void ShuffleGenerator::Generate(const PTX::_ShuffleInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void ShuffleGenerator::Visit(const PTX::ShuffleInstruction<T> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types: Bit32
	// Modifiers: --

	ArchitectureDispatch::Dispatch(*this, instruction);
}

template<class T>
void ShuffleGenerator::GenerateMaxwell(const PTX::ShuffleInstruction<T> *instruction)
{
	if constexpr(std::is_same<T, PTX::Bit32Type>::value)
	{
		// Generate operands

		RegisterGenerator registerGenerator(this->m_builder);
		CompositeGenerator compositeGenerator(this->m_builder);
		PredicateGenerator predicateGenerator(this->m_builder);

		auto destinationD = registerGenerator.Generate(instruction->GetDestination());

		auto destinationP_opt = instruction->GetDestinationP();
		auto destinationP = (destinationP_opt == nullptr) ? SASS::PT : predicateGenerator.Generate(destinationP_opt).first;

		auto sourceA = registerGenerator.Generate(instruction->GetSourceA());
		compositeGenerator.SetImmediateSize(8);
		auto sourceB = compositeGenerator.Generate(instruction->GetSourceB());
		compositeGenerator.SetImmediateSize(13);
		auto sourceC = compositeGenerator.Generate(instruction->GetSourceC());
		
		// Membermask is ignored for 6x targets as the membermask must be the same as activemask

		// Mode

		SASS::Maxwell::SHFLInstruction::ShuffleOperator shuffleOperator;
		switch (instruction->GetMode())
		{
			case PTX::ShuffleInstruction<T>::Mode::Up:
			{
				shuffleOperator = SASS::Maxwell::SHFLInstruction::ShuffleOperator::UP;
				break;
			}
			case PTX::ShuffleInstruction<T>::Mode::Down:
			{
				shuffleOperator = SASS::Maxwell::SHFLInstruction::ShuffleOperator::DOWN;
				break;
			}
			case PTX::ShuffleInstruction<T>::Mode::Butterfly:
			{
				shuffleOperator = SASS::Maxwell::SHFLInstruction::ShuffleOperator::BFLY;
				break;
			}
			case PTX::ShuffleInstruction<T>::Mode::Index:
			{
				shuffleOperator = SASS::Maxwell::SHFLInstruction::ShuffleOperator::IDX;
				break;
			}
		}

		// Generate instruction

		this->AddInstruction(new SASS::Maxwell::SHFLInstruction(
			destinationP, destinationD, sourceA, sourceB, sourceC, shuffleOperator
		));
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

template<class T>
void ShuffleGenerator::GenerateVolta(const PTX::ShuffleInstruction<T> *instruction)
{
	Error(instruction, "unsupported architecture");
}

}
}
