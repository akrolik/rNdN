#include "Backend/Codegen/Generators/Instructions/Data/UnpackGenerator.h"

#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void UnpackGenerator::Generate(const PTX::_UnpackInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T, PTX::VectorSize V>
void UnpackGenerator::Visit(const PTX::UnpackInstruction<T, V> *instruction)
{
	// Generate source register

	RegisterGenerator registerGenerator(this->m_builder);
	auto [source, sourceHi] = registerGenerator.Generate(instruction->GetSource());

	// Destination decomposition, split below

	auto destinations = instruction->GetDestination()->GetRegisters();

	//TODO: Instruction Unpack<T, V> types and vectors
	if constexpr(V == PTX::VectorSize::Vector2)
	{
		// Generate destination registers

		auto [destinationA, destinationA_Hi] = registerGenerator.Generate(destinations.at(0));
		auto [destinationB, destinationB_Hi] = registerGenerator.Generate(destinations.at(1));

		if constexpr(std::is_same<T, PTX::Bit16Type>::value)
		{
			auto temp0 = registerGenerator.GenerateTemporary(0);

			this->AddInstruction(new SASS::SHRInstruction(temp0, source, new SASS::I32Immediate(0x8)));
			this->AddInstruction(new SASS::LOPInstruction(
				destinationA, source, new SASS::I32Immediate(0xffff), SASS::LOPInstruction::BooleanOperator::AND
			));
			this->AddInstruction(new SASS::MOVInstruction(destinationB, temp0));
		}
		else if constexpr(std::is_same<T, PTX::Bit32Type>::value)
		{
		}
		else if constexpr(std::is_same<T, PTX::Bit64Type>::value)
		{
		}
	}
}

}
}
