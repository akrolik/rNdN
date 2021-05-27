#include "Backend/Codegen/Generators/Instructions/Arithmetic/CountLeadingZerosGenerator.h"

#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void CountLeadingZerosGenerator::Generate(const PTX::_CountLeadingZerosInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void CountLeadingZerosGenerator::Visit(const PTX::CountLeadingZerosInstruction<T> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types:
	//   - Bit32, Bit64    

	// Generate operands

	RegisterGenerator registerGenerator(this->m_builder);
	auto destination = registerGenerator.Generate(instruction->GetDestination());
	auto [source_Lo, source_Hi] = registerGenerator.GeneratePair(instruction->GetSource());

	if constexpr(std::is_same<T, PTX::Bit32Type>::value)
	{
		// FLO.U32 D, S ;
		// IADD32I D, -D, 0x1f ;

		this->AddInstruction(new SASS::FLOInstruction(destination, source_Lo));
		this->AddInstruction(new SASS::IADD32IInstruction(
			destination, destination, new SASS::I32Immediate(0x1f), SASS::IADD32IInstruction::Flags::NEG_A
		));
	}
	else if constexpr(std::is_same<T, PTX::Bit64Type>::value)
	{
		// ISETP.EQ.U32.AND P, PT, S_HI, RZ, PT ;
		// SEL TMP, RZ, 0x20, !P ;
		// SEL D, S, S_HI, P ;
		// FLO.U32 D, D ;
		// IADD32I D, -D, 0x1f ;
		// IADD D, TMP, D ;

		auto temp = this->m_builder.AllocateTemporaryRegister();
		auto predicate = this->m_builder.AllocateTemporaryPredicate();

		this->AddInstruction(new SASS::ISETPInstruction(
			predicate, SASS::PT, source_Hi, SASS::RZ, SASS::PT,
			SASS::ISETPInstruction::ComparisonOperator::EQ, SASS::ISETPInstruction::BooleanOperator::AND
		));

		this->AddInstruction(new SASS::SELInstruction(
			temp, SASS::RZ, new SASS::I32Immediate(0x20), predicate, SASS::SELInstruction::Flags::NOT_C
		));
		this->AddInstruction(new SASS::SELInstruction(destination, source_Lo, source_Hi, predicate));

		this->AddInstruction(new SASS::FLOInstruction(destination, destination));
		this->AddInstruction(new SASS::IADD32IInstruction(
			destination, destination, new SASS::I32Immediate(0x1f), SASS::IADD32IInstruction::Flags::NEG_A
		));

		this->AddInstruction(new SASS::IADDInstruction(destination, destination, temp));
	}
}

}
}
