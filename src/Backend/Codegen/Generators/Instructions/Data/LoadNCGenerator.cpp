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

template<class LDGInstruction, class T>
typename LDGInstruction::Type LoadNCGenerator::InstructionType()
{
	if constexpr(std::is_same<T, PTX::UInt8Type>::value)
	{
		return LDGInstruction::Type::U8;
	}
	else if constexpr(std::is_same<T, PTX::Int8Type>::value)
	{
		return LDGInstruction::Type::S8;
	}
	else if constexpr(std::is_same<T, PTX::UInt16Type>::value)
	{
		return LDGInstruction::Type::U16;
	}
	else if constexpr(std::is_same<T, PTX::Int16Type>::value)
	{
		return LDGInstruction::Type::S16;
	}
	else if constexpr(T::TypeBits == PTX::Bits::Bits32)
	{
		return LDGInstruction::Type::X32;
	}
	else if constexpr(T::TypeBits == PTX::Bits::Bits64)
	{
		return LDGInstruction::Type::X64;
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

	ArchitectureDispatch::DispatchInstruction<
		SASS::Maxwell::LDGInstruction,
		SASS::Volta::LDGInstruction
	>(*this, instruction);
}

template<class LDGInstruction, PTX::Bits B, class T, class S>
void LoadNCGenerator::GenerateInstruction(const PTX::LoadNCInstruction<B, T, S> *instruction)
{
	if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
	{
		// Generate operands

		RegisterGenerator registerGenerator(this->m_builder);
		auto destination = registerGenerator.Generate(instruction->GetDestination());

		AddressGenerator addressGenerator(this->m_builder);
		auto address = addressGenerator.Generate(instruction->GetAddress());

		// Generate instruction

		auto type = InstructionType<LDGInstruction, T>();
		auto flags = LDGInstruction::Flags::None;
		if constexpr(B == PTX::Bits::Bits64)
		{
			flags = LDGInstruction::Flags::E;
		}

		// Cache depends on the instruction kind
		
		if constexpr(std::is_same<LDGInstruction, SASS::Volta::LDGInstruction>::value)
		{
			auto cache = LDGInstruction::Cache::CONSTANT;
			auto evict = LDGInstruction::Evict::None;
			auto prefetch = LDGInstruction::Prefetch::None;

			this->AddInstruction(new LDGInstruction(destination, address, type, cache, evict, prefetch, flags));
		}
		else
		{
			auto cache = LDGInstruction::Cache::CI;

			this->AddInstruction(new LDGInstruction(destination, address, type, cache, flags));
		}

	}
	else
	{
		Error(instruction, "unsupported space");
	}
}

}
}
