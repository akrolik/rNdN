#include "Backend/Codegen/Generators/Instructions/Arithmetic/RemainderGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Utils/Math.h"

namespace Backend {
namespace Codegen {

void RemainderGenerator::Generate(const PTX::_RemainderInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void RemainderGenerator::Visit(const PTX::RemainderInstruction<T> *instruction)
{
	// Types:
	//   - Int16, Int32, Int64
	//   - UInt16, UInt32, UInt64
	// Modifiers: --

	// Generate operands

	RegisterGenerator registerGenerator(this->m_builder);
	CompositeGenerator compositeGenerator(this->m_builder);

	auto [destination, destination_Hi] = registerGenerator.Generate(instruction->GetDestination());
	auto [sourceA, sourceA_Hi] = registerGenerator.Generate(instruction->GetSourceA());
	auto [sourceB, sourceB_Hi] = compositeGenerator.Generate(instruction->GetSourceB());

	// Generate instruction

	//TODO: Instruction Remainder<T> types
	if constexpr(std::is_same<T, PTX::UInt32Type>::value)
	{
		// Optimize power of 2 remainder using bitwise &(divisor-1)

		if (auto immediateSourceB = dynamic_cast<SASS::I32Immediate *>(sourceB))
		{
			auto value = immediateSourceB->GetValue();
			if (value == Utils::Math::Power2(value))
			{
				immediateSourceB->SetValue(value - 1);
				this->AddInstruction(new SASS::LOPInstruction(
					destination, sourceA, immediateSourceB, SASS::LOPInstruction::BooleanOperator::AND
				));
			}
		}
	}
}

}
}
