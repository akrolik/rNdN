#include "Backend/Codegen/Generators/Instructions/Data/StoreGenerator.h"

#include "Backend/Codegen/Generators/Operands/AddressGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void StoreGenerator::Generate(const PTX::_StoreInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
SASS::STGInstruction::Type StoreGenerator::InstructionType()
{
	if constexpr(std::is_same<T, PTX::UInt8Type>::value)
	{
		return SASS::STGInstruction::Type::U8;
	}
	else if constexpr(std::is_same<T, PTX::Int8Type>::value)
	{
		return SASS::STGInstruction::Type::S8;
	}
	else if constexpr(std::is_same<T, PTX::UInt16Type>::value)
	{
		return SASS::STGInstruction::Type::U16;
	}
	else if constexpr(std::is_same<T, PTX::Int16Type>::value)
	{
		return SASS::STGInstruction::Type::S16;
	}
	else if constexpr(T::TypeBits == PTX::Bits::Bits32)
	{
		return SASS::STGInstruction::Type::X32;
	}
	else if constexpr(T::TypeBits == PTX::Bits::Bits64)
	{
		return SASS::STGInstruction::Type::X64;
	}
	Error("store for type " + T::Name());
}

template<PTX::Bits B, class T, class S, PTX::StoreSynchronization A>
void StoreGenerator::Visit(const PTX::StoreInstruction<B, T, S, A> *instruction)
{
	// Types: *, Vector (exclude Float16(x2))
	// Spaces: *
	// Modifiers: --

	// Generate source operand

	RegisterGenerator registerGenerator(this->m_builder);
	auto [source, source_Hi] = registerGenerator.Generate(instruction->GetSource());

	//TODO: Instruction Store<T> types, modifiers, atomics
	if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
	{
		// Generate address operand

		AddressGenerator addressGenerator(this->m_builder);
		auto address = addressGenerator.Generate(instruction->GetAddress());

		// Generate instruction

		auto type = InstructionType<T>();
		auto flags = SASS::STGInstruction::Flags::None;
		if constexpr(B == PTX::Bits::Bits64)
		{
			flags = SASS::STGInstruction::Flags::E;
		}
		auto cache = SASS::STGInstruction::Cache::None;

		this->AddInstruction(new SASS::STGInstruction(address, source, type, cache, flags));
	}
}

}
}
