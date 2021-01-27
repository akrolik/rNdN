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
	// Generate source and destination registers (the source is always a register address)

	RegisterGenerator registerGenerator(this->m_builder);
	auto [destination, destinationHi] = registerGenerator.Generate(instruction->GetDestination());
	auto [source, sourceHi] = registerGenerator.Generate(instruction->GetAddress()->GetRegister());

	// Move address from source to destination

	this->AddInstruction(new SASS::MOVInstruction(destination, source));
	if (destinationHi != nullptr && sourceHi != nullptr)
	{
		this->AddInstruction(new SASS::MOVInstruction(destinationHi, sourceHi));
	}
}

}
}
