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
	// Types:
	//   - Bit16, Bit32, Bit64
	// Vector
	//   - Vector2
	//   - Vector4

	// Generate source register

	RegisterGenerator registerGenerator(this->m_builder);
	auto [source, source_Hi] = registerGenerator.Generate(instruction->GetSource());

	// Destination decomposition, split below

	auto destinations = instruction->GetDestination()->GetRegisters();

	// Generate instruction

	if constexpr(V == PTX::VectorSize::Vector2)
	{
		// Generate destination registers

		auto [destinationA, destinationA_Hi] = registerGenerator.Generate(destinations.at(0));
		auto [destinationB, destinationB_Hi] = registerGenerator.Generate(destinations.at(1));

		// Temporary necessary for register reuse

		auto temp0 = registerGenerator.GenerateTemporary(0);

		if constexpr(std::is_same<T, PTX::Bit16Type>::value)
		{
			this->AddInstruction(new SASS::SHRInstruction(temp0, source, new SASS::I32Immediate(0x8)));
			this->AddInstruction(new SASS::LOPInstruction(
				destinationA, source, new SASS::I32Immediate(0xffff), SASS::LOPInstruction::BooleanOperator::AND
			));
			this->AddInstruction(new SASS::MOVInstruction(destinationB, temp0));
		}
		else if constexpr(std::is_same<T, PTX::Bit32Type>::value)
		{
			//TODO: UnpackInstruction Vector2<Bit32Type>
		}
		else if constexpr(std::is_same<T, PTX::Bit64Type>::value)
		{
			this->AddInstruction(new SASS::MOVInstruction(temp0, source));
			this->AddInstruction(new SASS::MOVInstruction(destinationB, source_Hi));
			this->AddInstruction(new SASS::MOVInstruction(destinationA, temp0));
		}
	}
	else if constexpr(V == PTX::VectorSize::Vector4)
	{
		//TODO: UnpackInstruction Vector4<Bit16Type/Bit32Type/Bit64Type>
	}
}

}
}
