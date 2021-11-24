#include "Backend/Codegen/Generators/Instructions/Comparison/SelectGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void SelectGenerator::Generate(const PTX::_SelectInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void SelectGenerator::Visit(const PTX::SelectInstruction<T> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types:
	//   - Bit16, Bit32, Bit64
	//   - Int16, Int32, Int64
	//   - UInt16, UInt32, UInt64
	//   - Float32, Float64
	// Modifiers: --

	ArchitectureDispatch::DispatchInstruction<
		SASS::Maxwell::SELInstruction,
		SASS::Volta::SELInstruction
	>(*this, instruction);
}

template<class SELInstruction, class T>
void SelectGenerator::GenerateInstruction(const PTX::SelectInstruction<T> *instruction)
{
	// Generate operands

	RegisterGenerator registerGenerator(this->m_builder);
	CompositeGenerator compositeGenerator(this->m_builder);

	if constexpr(std::is_same<SELInstruction, SASS::Volta::SELInstruction>::value)
	{
		compositeGenerator.SetImmediateSize(32);
	}

	auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());
	auto [sourceA_Lo, sourceA_Hi] = registerGenerator.GeneratePair(instruction->GetSourceA());
	auto [sourceB_Lo, sourceB_Hi] = compositeGenerator.GeneratePair(instruction->GetSourceB());

	PredicateGenerator predicateGenerator(this->m_builder);
	auto [sourceC, sourceC_Not] = predicateGenerator.Generate(instruction->GetSourceC());

	// Flags

	auto flags = SELInstruction::Flags::None;
	if (sourceC_Not)
	{
		flags |= SELInstruction::Flags::NOT_C;
	}

	// Generate instruction

	this->AddInstruction(new SELInstruction(destination_Lo, sourceA_Lo, sourceB_Lo, sourceC, flags));
	if constexpr(T::TypeBits == PTX::Bits::Bits64)
	{
		this->AddInstruction(new SELInstruction(destination_Hi, sourceA_Hi, sourceB_Hi, sourceC, flags));
	}
}

}
}
