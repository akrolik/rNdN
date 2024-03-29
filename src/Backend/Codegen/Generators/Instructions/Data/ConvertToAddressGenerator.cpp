#include "Backend/Codegen/Generators/Instructions/Data/ConvertToAddressGenerator.h"

#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void ConvertToAddressGenerator::Generate(const PTX::_ConvertToAddressInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<PTX::Bits B, class T, class S>
void ConvertToAddressGenerator::Visit(const PTX::ConvertToAddressInstruction<B, T, S> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types: *
	// Spaces: Addressable subspace

	ArchitectureDispatch::DispatchInstruction<
		SASS::Maxwell::MOVInstruction, SASS::Volta::MOVInstruction
	>(*this, instruction);
}

template<class MOVInstruction, PTX::Bits B, class T, class S>
void ConvertToAddressGenerator::GenerateInstruction(const PTX::ConvertToAddressInstruction<B, T, S> *instruction)
{
	// Generate source and destination registers (the source is always a register address)

	RegisterGenerator registerGenerator(this->m_builder);
	auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());
	auto [source_Lo, source_Hi] = registerGenerator.GeneratePair(instruction->GetAddress()->GetRegister());

	// Move address from source to destination

	this->AddInstruction(new MOVInstruction(destination_Lo, source_Lo));
	if constexpr(B == PTX::Bits::Bits64)
	{
		this->AddInstruction(new MOVInstruction(destination_Hi, source_Hi));
	}
}

}
}
