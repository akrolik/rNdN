#include "Backend/Codegen/Generators/Instructions/Arithmetic/MultiplyGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void MultiplyGenerator::Generate(const PTX::_MultiplyInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void MultiplyGenerator::Visit(const PTX::MultiplyInstruction<T> *instruction)
{
	// Types:
	//   - Int16, Int32, Int64
	//   - UInt16, UInt32, UInt64
	//   - Float16, Float16x2, Float32, Float64
	// Modifiers:
	//   - Half: Int16, Int32, Int64, UInt16, UInt32, UInt64
	//   - FlushSubnormal: Float16, Float16x2, Float32
	//   - Rounding: Float16, Float16x2, Float32, Float64
	//   - Saturate: Int32, Float16, Float16x2, Float32

	// Generate operands

	RegisterGenerator registerGenerator(this->m_builder);
	CompositeGenerator compositeGenerator(this->m_builder);

	auto [destination, destination_Hi] = registerGenerator.Generate(instruction->GetDestination());
	auto [sourceA, sourceA_Hi] = registerGenerator.Generate(instruction->GetSourceA());
	auto [sourceB, sourceB_Hi] = compositeGenerator.Generate(instruction->GetSourceB());

	// Generate instruction

	//TODO: Instruction Multiply<T> types/modifiers
	if constexpr(std::is_same<T, PTX::Float64Type>::value)
	{
		this->AddInstruction(new SASS::DMULInstruction(destination, sourceA, sourceB));
	}
}

}
}
