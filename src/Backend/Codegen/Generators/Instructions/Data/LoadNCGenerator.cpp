#include "Backend/Codegen/Generators/Instructions/Data/LoadNCGenerator.h"

#include "Backend/Codegen/Generators/Operands/AddressGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void LoadNCGenerator::Generate(const PTX::_LoadNCInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class I, class T>
typename I::Type LoadNCGenerator::InstructionType()
{
	if constexpr(std::is_same<T, PTX::UInt8Type>::value)
	{
		return I::Type::U8;
	}
	else if constexpr(std::is_same<T, PTX::Int8Type>::value)
	{
		return I::Type::S8;
	}
	else if constexpr(std::is_same<T, PTX::UInt16Type>::value)
	{
		return I::Type::U16;
	}
	else if constexpr(std::is_same<T, PTX::Int16Type>::value)
	{
		return I::Type::S16;
	}
	else if constexpr(T::TypeBits == PTX::Bits::Bits32)
	{
		return I::Type::X32;
	}
	else if constexpr(T::TypeBits == PTX::Bits::Bits64)
	{
		return I::Type::X64;
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

	ArchitectureDispatch::Dispatch(*this, instruction);
}

template<PTX::Bits B, class T, class S>
void LoadNCGenerator::GenerateMaxwell(const PTX::LoadNCInstruction<B, T, S> *instruction)
{
	if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
	{
		// Generate operands

		RegisterGenerator registerGenerator(this->m_builder);
		auto destination = registerGenerator.Generate(instruction->GetDestination());

		AddressGenerator addressGenerator(this->m_builder);
		auto address = addressGenerator.Generate(instruction->GetAddress());

		// Generate instruction

		auto type = InstructionType<SASS::Maxwell::LDGInstruction, T>();
		auto flags = SASS::Maxwell::LDGInstruction::Flags::None;
		if constexpr(B == PTX::Bits::Bits64)
		{
			flags = SASS::Maxwell::LDGInstruction::Flags::E;
		}
		auto cache = SASS::Maxwell::LDGInstruction::Cache::CI;

		this->AddInstruction(new SASS::Maxwell::LDGInstruction(destination, address, type, cache, flags));
	}
	else
	{
		Error(instruction, "unsupported space");
	}
}

template<PTX::Bits B, class T, class S>
void LoadNCGenerator::GenerateVolta(const PTX::LoadNCInstruction<B, T, S> *instruction)
{
	Error(instruction, "unsupported architecture");
}

}
}
