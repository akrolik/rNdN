#include "Backend/Codegen/Generators/Instructions/Data/ConvertToAddressGenerator.h"

#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void ConvertToAddressGenerator::Generate(const PTX::_ConvertToAddressInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<PTX::Bits B, class T, class S>
void ConvertToAddressGenerator::Visit(const PTX::ConvertToAddressInstruction<B, T, S> *instruction)
{
	// Types: *
	// Spaces: Addressable subspace

	// Generate source and destination registers (the source is always a register address)

	RegisterGenerator registerGenerator(this->m_builder);
	auto [destination, destination_Hi] = registerGenerator.Generate(instruction->GetDestination());
	auto [source, source_Hi] = registerGenerator.Generate(instruction->GetAddress()->GetRegister());

	// Move address from source to destination

	this->AddInstruction(new SASS::MOVInstruction(destination, source));
	if constexpr(B == PTX::Bits::Bits64)
	{
		this->AddInstruction(new SASS::MOVInstruction(destination_Hi, source_Hi));
	}
}

}
}
