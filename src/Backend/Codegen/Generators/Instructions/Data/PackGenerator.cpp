#include "Backend/Codegen/Generators/Instructions/Data/PackGenerator.h"

#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void PackGenerator::Generate(const PTX::_PackInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T, PTX::VectorSize V>
void PackGenerator::Visit(const PTX::PackInstruction<T, V> *instruction)
{
	// Types:
	//   - Bit16, Bit32, Bit64
	// Vector
	//   - Vector2
	//   - Vector4

	// Generate destination register

	RegisterGenerator registerGenerator(this->m_builder);
	auto [destination, destination_Hi] = registerGenerator.Generate(instruction->GetDestination());

	// Source decomposition, split below

	auto sources = instruction->GetSource()->GetOperands();

	// Generate instruction

	if constexpr(V == PTX::VectorSize::Vector2)
	{
		auto [sourceA, sourceA_Hi] = registerGenerator.Generate(sources.at(0));
		auto [sourceB, sourceB_Hi] = registerGenerator.Generate(sources.at(1));

		// Temporary necessary for register reuse

		auto temp0 = registerGenerator.GenerateTemporary(0);

		if constexpr(std::is_same<T, PTX::Bit16Type>::value)
		{
			this->AddInstruction(new SASS::SHLInstruction(temp0, sourceB, new SASS::I32Immediate(0x8)));
			this->AddInstruction(new SASS::LOPInstruction(destination, sourceA, temp0, SASS::LOPInstruction::BooleanOperator::OR));
		}
		else if constexpr(std::is_same<T, PTX::Bit32Type>::value)
		{
			//TODO: PackInstruction Vector2<Bit32Type>
		}
		else if constexpr(std::is_same<T, PTX::Bit64Type>::value)
		{
			this->AddInstruction(new SASS::MOVInstruction(temp0, sourceA));
			this->AddInstruction(new SASS::MOVInstruction(destination_Hi, sourceB));
			this->AddInstruction(new SASS::MOVInstruction(destination, temp0));
		}
	}
	else if constexpr(V == PTX::VectorSize::Vector4)
	{
		//TODO: PackInstruction Vector4<Bit16Type/Bit32Type/Bit64Type>
	}
}

}
}
