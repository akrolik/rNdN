#include "Backend/Codegen/Generators/Instructions/Data/LoadNCGenerator.h"

#include "Backend/Codegen/Generators/Operands/AddressGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void LoadNCGenerator::Generate(const PTX::_LoadNCInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
SASS::LDGInstruction::Type LoadNCGenerator::InstructionType()
{
	if constexpr(std::is_same<T, PTX::UInt8Type>::value)
	{
		return SASS::LDGInstruction::Type::U8;
	}
	else if constexpr(std::is_same<T, PTX::Int8Type>::value)
	{
		return SASS::LDGInstruction::Type::S8;
	}
	else if constexpr(std::is_same<T, PTX::UInt16Type>::value)
	{
		return SASS::LDGInstruction::Type::U16;
	}
	else if constexpr(std::is_same<T, PTX::Int16Type>::value)
	{
		return SASS::LDGInstruction::Type::S16;
	}
	else if constexpr(T::TypeBits == PTX::Bits::Bits32)
	{
		return SASS::LDGInstruction::Type::X32;
	}
	else if constexpr(T::TypeBits == PTX::Bits::Bits64)
	{
		return SASS::LDGInstruction::Type::X64;
	}
	Error("load.nc for type " + T::Name());
}

template<PTX::Bits B, class T, class S>
void LoadNCGenerator::Visit(const PTX::LoadNCInstruction<B, T, S> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types: *, Vector (exclude Float16(x2))
	// Spaces: Global
	// Modifiers: --

	if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
	{
		// Generate operands

		RegisterGenerator registerGenerator(this->m_builder);
		auto destination = registerGenerator.Generate(instruction->GetDestination());

		AddressGenerator addressGenerator(this->m_builder);
		auto address = addressGenerator.Generate(instruction->GetAddress());

		// Generate instruction

		auto type = InstructionType<T>();
		auto flags = SASS::LDGInstruction::Flags::None;
		if constexpr(B == PTX::Bits::Bits64)
		{
			flags = SASS::LDGInstruction::Flags::E;
		}
		auto cache = SASS::LDGInstruction::Cache::CI;

		this->AddInstruction(new SASS::LDGInstruction(destination, address, type, cache, flags));
	}
	else
	{
		Error(instruction, "unsupported space");
	}
}

}
}
