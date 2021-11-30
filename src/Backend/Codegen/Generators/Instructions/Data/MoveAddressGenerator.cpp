#include "Backend/Codegen/Generators/Instructions/Data/MoveAddressGenerator.h"

#include "Backend/Codegen/Generators/Operands/AddressGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void MoveAddressGenerator::Generate(const PTX::_MoveAddressInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<PTX::Bits B, class T, class S>
void MoveAddressGenerator::Visit(const PTX::MoveAddressInstruction<B, T, S> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	ArchitectureDispatch::DispatchInstruction<
		SASS::Maxwell::MOVInstruction, SASS::Volta::MOVInstruction
	>(*this, instruction);
}

template<class MOVInstruction, PTX::Bits B, class T, class S>
void MoveAddressGenerator::GenerateInstruction(const PTX::MoveAddressInstruction<B, T, S> *instruction)
{
	// Generate operands

	RegisterGenerator registerGenerator(this->m_builder);
	auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());

	AddressGenerator addressGenerator(this->m_builder);
	addressGenerator.SetUseOffset(false); // Add offset to base
	auto address = addressGenerator.Generate(instruction->GetAddress());

	// Generate instruction (no overlap unless equal)

	auto addressRegister_Lo = address->GetBase();
	this->AddInstruction(new MOVInstruction(destination_Lo, addressRegister_Lo));

	if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
	{
		if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			if (addressRegister_Lo->GetValue() == SASS::Register::ZeroIndex)
			{
				this->AddInstruction(new MOVInstruction(destination_Hi, SASS::RZ));
			}
			else
			{
				auto addressRegister_Hi = new SASS::Register(addressRegister_Lo->GetValue() + 1);
				this->AddInstruction(new MOVInstruction(destination_Hi, addressRegister_Hi));
			}
		}
	}
	else if constexpr(!std::is_same<S, PTX::SharedSpace>::value)
	{
		Error(instruction, "unsupported space");
	}
}

}
}
