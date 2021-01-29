#include "Backend/Codegen/Generators/Instructions/Comparison/SelectGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void SelectGenerator::Generate(const PTX::_SelectInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void SelectGenerator::Visit(const PTX::SelectInstruction<T> *instruction)
{
	// Types:
	//   - Bit16, Bit32, Bit64
	//   - Int16, Int32, Int64
	//   - UInt16, UInt32, UInt64
	//   - Float32, Float64
	// Modifiers: --

	// Generate operands

	RegisterGenerator registerGenerator(this->m_builder);
	CompositeGenerator compositeGenerator(this->m_builder);

	auto [destination, destination_Hi] = registerGenerator.Generate(instruction->GetDestination());
	auto [sourceA, sourceA_Hi] = registerGenerator.Generate(instruction->GetSourceA());
	auto [sourceB, sourceB_Hi] = compositeGenerator.Generate(instruction->GetSourceB());

	PredicateGenerator predicateGenerator(this->m_builder);
	auto sourceC = predicateGenerator.Generate(instruction->GetSourceC());

	// Generate instruction

	this->AddInstruction(new SASS::SELInstruction(destination, sourceA, sourceB, sourceC));
	if constexpr(T::TypeBits == PTX::Bits::Bits64)
	{
		this->AddInstruction(new SASS::SELInstruction(destination_Hi, sourceA_Hi, sourceB_Hi, sourceC));
	}
}

}
}
