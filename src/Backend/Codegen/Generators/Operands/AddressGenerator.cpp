#include "Backend/Codegen/Generators/Operands/AddressGenerator.h"

#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "PTX/Utils/PrettyPrinter.h"

namespace Backend {
namespace Codegen {

SASS::Address *AddressGenerator::Generate(const PTX::Operand *operand)
{
	// Clear

	m_address = nullptr;

	// Generate address

	operand->Accept(*this);
	if (m_address == nullptr)
	{
		Error(operand, "unsupported kind");
	}
	return m_address;
}

void AddressGenerator::Visit(const PTX::_MemoryAddress *address)
{
	address->Dispatch(*this);
}

void AddressGenerator::Visit(const PTX::_RegisterAddress *address)
{
	address->Dispatch(*this);
}

template<PTX::Bits B, class T, class S>
void AddressGenerator::Visit(const PTX::MemoryAddress<B, T, S> *address)
{
	const auto& name = address->GetVariable()->GetName();
	if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
	{
		// Global addresses are initially zero and relocated at runtime

		auto [temp_Lo, temp_Hi] = this->m_builder.AllocateTemporaryRegisterPair<B>();
		auto inst0 = new SASS::MOV32IInstruction(temp_Lo, new SASS::I32Immediate(0x0));

		this->m_builder.AddInstruction(inst0);
		this->m_builder.AddRelocation(inst0, name, SASS::Relocation::Kind::ABS32_LO_20);

		// Extended addresses (64-bit)

		if constexpr(B == PTX::Bits::Bits64)
		{
			auto inst1 = new SASS::MOV32IInstruction(temp_Hi, new SASS::I32Immediate(0x0));

			this->m_builder.AddInstruction(inst1);
			this->m_builder.AddRelocation(inst1, name, SASS::Relocation::Kind::ABS32_HI_20);
		}

		// Add offset to base

		if (auto addressOffset = address->GetOffset() * static_cast<int>(sizeof(typename T::SystemType)))
		{
			this->m_builder.AddInstruction(new SASS::IADD32IInstruction(
				temp_Lo, temp_Lo, new SASS::I32Immediate(addressOffset), SASS::IADD32IInstruction::Flags::CC
			));

			if constexpr(B == PTX::Bits::Bits64)
			{
				this->m_builder.AddInstruction(new SASS::IADD32IInstruction(
					temp_Hi, temp_Hi, new SASS::I32Immediate(0x0), SASS::IADD32IInstruction::Flags::X
				));
			}
		}

		// Form the address

		auto size = Utils::Math::DivUp(PTX::BitSize<B>::NumBits, 32);
		auto temp = new SASS::Register(temp_Lo->GetValue(), size);

		m_address = new SASS::Address(temp);
	}
	else if constexpr(std::is_same<S, PTX::SharedSpace>::value)
	{
		// Shared addresses are initially zero and relocated at runtime

		auto temp = this->m_builder.AllocateTemporaryRegister();
		auto inst = new SASS::MOV32IInstruction(temp, new SASS::I32Immediate(0x0));

		this->m_builder.AddInstruction(inst);
		this->m_builder.AddRelocation(inst, name, SASS::Relocation::Kind::ABS24_20);

		// Form the address with the offset

		if (auto addressOffset = address->GetOffset() * static_cast<int>(sizeof(typename T::SystemType)))
		{
			this->m_builder.AddInstruction(new SASS::IADD32IInstruction(temp, temp, new SASS::I32Immediate(addressOffset)));
		}
		m_address = new SASS::Address(temp);
	}
	else
	{
		Error(address, "unsupported space");
	}
}

template<PTX::Bits B, class T, class S>
void AddressGenerator::Visit(const PTX::RegisterAddress<B, T, S> *address)
{
	// Generate register for address

	RegisterGenerator registerGenerator(this->m_builder);
	auto [reg, regHi] = registerGenerator.GeneratePair(address->GetRegister());

	// Construct address with offset

	if (auto addressOffset = address->GetOffset() * static_cast<int>(sizeof(typename T::SystemType)))
	{
		if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
		{
			auto [temp_Lo, temp_Hi] = this->m_builder.AllocateTemporaryRegisterPair<B>();

			this->m_builder.AddInstruction(new SASS::IADD32IInstruction(
				temp_Lo, reg, new SASS::I32Immediate(addressOffset), SASS::IADD32IInstruction::Flags::CC
			));

			if constexpr(B == PTX::Bits::Bits64)
			{
				this->m_builder.AddInstruction(new SASS::IADD32IInstruction(
					temp_Hi, regHi, new SASS::I32Immediate(0x0), SASS::IADD32IInstruction::Flags::X
				));
			}

			auto size = Utils::Math::DivUp(PTX::BitSize<B>::NumBits, 32);
			auto temp = new SASS::Register(temp_Lo->GetValue(), size);

			m_address = new SASS::Address(temp);
		}
		else if constexpr(std::is_same<S, PTX::SharedSpace>::value)
		{
			auto temp = this->m_builder.AllocateTemporaryRegister();

			this->m_builder.AddInstruction(new SASS::IADD32IInstruction(temp, reg, new SASS::I32Immediate(addressOffset)));

			m_address = new SASS::Address(temp);
		}
		else
		{
			Error(address, "unsupported space");
		}
	}
	else
	{
		auto size = Utils::Math::DivUp(PTX::BitSize<B>::NumBits, 32);
		auto temp = new SASS::Register(reg->GetValue(), size);

		m_address = new SASS::Address(temp);
	}
}

}
}
