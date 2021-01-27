#include "Backend/Codegen/Generators/Instructions/Data/LoadGenerator.h"

#include "Backend/Codegen/Generators/Operands/AddressGenerator.h"
#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void LoadGenerator::Generate(const PTX::_LoadInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
SASS::LDGInstruction::Type LoadGenerator::InstructionType()
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
		return SASS::LDGInstruction::Type::I32;
	}
	else if constexpr(T::TypeBits == PTX::Bits::Bits64)
	{
		return SASS::LDGInstruction::Type::I64;
	}
	Error("load.nc for type " + T::Name());
}

template<PTX::Bits B, class T, class S, PTX::LoadSynchronization A>
void LoadGenerator::Visit(const PTX::LoadInstruction<B, T, S, A> *instruction)
{
	// Generate destination operand

	RegisterGenerator registerGenerator(this->m_builder);
	auto [destination, destinationHi] = registerGenerator.Generate(instruction->GetDestination());

	//TODO: Instruction Load<T> types, modifiers, spaces, atomics
	if constexpr(std::is_same<S, PTX::ParameterSpace>::value)
	{
		CompositeGenerator compositeGenerator(this->m_builder);
		auto [address, addressHi] = compositeGenerator.Generate(instruction->GetAddress());

		// Generate instruction

		this->AddInstruction(new SASS::MOVInstruction(destination, address));

		// Extended datatypes

		if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			this->AddInstruction(new SASS::MOVInstruction(destinationHi, addressHi));
		}
	}
	else if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
	{
		// Generate address operand

		AddressGenerator addressGenerator(this->m_builder);
		auto address = addressGenerator.Generate(instruction->GetAddress());

		// Generate instruction

		auto type = InstructionType<T>();
		auto flags = SASS::LDGInstruction::Flags::None;
		if constexpr(B == PTX::Bits::Bits64)
		{
			flags = SASS::LDGInstruction::Flags::E;
		}
		auto cache = SASS::LDGInstruction::Cache::None;

		this->AddInstruction(new SASS::LDGInstruction(destination, address, type, cache, flags));
		this->AddInstruction(new SASS::DEPBARInstruction(SASS::DEPBARInstruction::Barrier::SB0, new SASS::I8Immediate(0x0), SASS::DEPBARInstruction::Flags::LE));
	}
}

}
}
