#include "Backend/Codegen/Generators/Operands/AddressGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"
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

bool AddressGenerator::Visit(const PTX::_MemoryAddress *address)
{
	address->Dispatch(*this);
	return false;
}

bool AddressGenerator::Visit(const PTX::_RegisterAddress *address)
{
	address->Dispatch(*this);
	return false;
}

template<PTX::Bits B, class T, class S>
void AddressGenerator::Visit(const PTX::MemoryAddress<B, T, S> *address)
{
	const auto& name = address->GetVariable()->GetName();
	auto addressOffset = address->GetOffset() * static_cast<int>(sizeof(typename T::SystemType));

	if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
	{
		// Global addresses are initially zero and relocated at runtime

		auto [temp_Lo, temp_Hi] = this->m_builder.AllocateTemporaryRegisterPair<B>();

		ArchitectureDispatch::DispatchInline(m_builder,
		[&]() // Maxwell instruction set
		{
			auto inst0 = new SASS::Maxwell::MOV32IInstruction(temp_Lo, new SASS::I32Immediate(0x0));

			this->m_builder.AddInstruction(inst0);
			this->m_builder.AddRelocation(inst0, name, SASS::Relocation::Kind::ABS32_LO_20);

			// Extended addresses (64-bit)

			if constexpr(B == PTX::Bits::Bits64)
			{
				auto inst1 = new SASS::Maxwell::MOV32IInstruction(temp_Hi, new SASS::I32Immediate(0x0));

				this->m_builder.AddInstruction(inst1);
				this->m_builder.AddRelocation(inst1, name, SASS::Relocation::Kind::ABS32_HI_20);
			}
		},
		[&]() // Volta instruction set
		{
			auto inst0 = new SASS::Volta::MOVInstruction(temp_Lo, new SASS::I32Immediate(0x0));

			this->m_builder.AddInstruction(inst0);
			this->m_builder.AddRelocation(inst0, name, SASS::Relocation::Kind::ABS32_LO_32);

			// Extended addresses (64-bit)

			if constexpr(B == PTX::Bits::Bits64)
			{
				auto inst1 = new SASS::Volta::MOVInstruction(temp_Hi, new SASS::I32Immediate(0x0));

				this->m_builder.AddInstruction(inst1);
				this->m_builder.AddRelocation(inst1, name, SASS::Relocation::Kind::ABS32_HI_32);
			}
		});

		// Form the address with the offset

		auto size = Utils::Math::DivUp(PTX::BitSize<B>::NumBits, 32);
		auto temp = new SASS::Register(temp_Lo->GetValue(), size);

		if (addressOffset >= (1 << 24) || !m_useOffset)
		{
			auto addressImmediate = new SASS::I32Immediate(addressOffset);

			ArchitectureDispatch::DispatchInline(m_builder,
			[&]() // Maxwell instruction set
			{
				this->m_builder.AddInstruction(new SASS::Maxwell::IADD32IInstruction(
					temp_Lo, temp_Lo, addressImmediate, SASS::Maxwell::IADD32IInstruction::Flags::CC
				));

				if constexpr(B == PTX::Bits::Bits64)
				{
					this->m_builder.AddInstruction(new SASS::Maxwell::IADD32IInstruction(
						temp_Hi, temp_Hi, new SASS::I32Immediate(0x0), SASS::Maxwell::IADD32IInstruction::Flags::X
					));
				}
			},
			[&]() // Volta instruction set
			{
				if constexpr(B == PTX::Bits::Bits32)
				{
					this->m_builder.AddInstruction(new SASS::Volta::IADD3Instruction(
						temp_Lo, temp_Lo, addressImmediate, SASS::RZ
					));
				}
				else
				{
					auto CC = this->m_builder.AllocateTemporaryPredicate();

					this->m_builder.AddInstruction(new SASS::Volta::IADD3Instruction(
						temp_Lo, CC, temp_Lo, addressImmediate, SASS::RZ
					));

					this->m_builder.AddInstruction(new SASS::Volta::IADD3Instruction(
						temp_Hi, temp_Hi, SASS::RZ, SASS::RZ, CC, SASS::PT, SASS::Volta::IADD3Instruction::Flags::NOT_E
					));
				}
			});

			m_address = new SASS::Address(temp);
		}
		else
		{
			m_address = new SASS::Address(temp, addressOffset);
		}
	}
	else if constexpr(std::is_same<S, PTX::SharedSpace>::value)
	{
		// Shared addresses are initially zero and relocated at runtime

		auto temp = this->m_builder.AllocateTemporaryRegister();

		ArchitectureDispatch::DispatchInline(m_builder,
		[&]() // Maxwell instruction set
		{
			auto inst = new SASS::Maxwell::MOV32IInstruction(temp, new SASS::I32Immediate(0x0));

			this->m_builder.AddInstruction(inst);
			this->m_builder.AddRelocation(inst, name, SASS::Relocation::Kind::ABS24_20);
		},
		[&]() // Volta instruction set
		{
			auto inst = new SASS::Volta::MOVInstruction(temp, new SASS::I32Immediate(0x0));

			this->m_builder.AddInstruction(inst);
			this->m_builder.AddRelocation(inst, name, SASS::Relocation::Kind::ABS32_32);
		});

		// Form the address with the offset

		if (addressOffset >= (1 << 24) || !m_useOffset)
		{
			auto addressImmediate = new SASS::I32Immediate(addressOffset);

			ArchitectureDispatch::DispatchInline(m_builder,
			[&]() // Maxwell instruction set
			{
				this->m_builder.AddInstruction(new SASS::Maxwell::IADD32IInstruction(temp, temp, addressImmediate));
			},
			[&]() // Volta instruction set
			{
				this->m_builder.AddInstruction(new SASS::Volta::IADD3Instruction(temp, temp, addressImmediate, SASS::RZ));
			});

			m_address = new SASS::Address(temp);
		}
		else
		{
			m_address = new SASS::Address(temp, addressOffset);
		}
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
	auto [reg_Lo, reg_Hi] = registerGenerator.GeneratePair(address->GetRegister());

	// Construct address with offset

	auto addressOffset = address->GetOffset() * static_cast<int>(sizeof(typename T::SystemType));

	if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
	{
		auto size = Utils::Math::DivUp(PTX::BitSize<B>::NumBits, 32);

		// Inline offset has maximum size

		if (addressOffset >= (1 << 24) || !m_useOffset)
		{
			auto [temp_Lo, temp_Hi] = this->m_builder.AllocateTemporaryRegisterPair<B>();
			auto addressImmediate = new SASS::I32Immediate(addressOffset);

			ArchitectureDispatch::DispatchInline(m_builder,
			[&]() // Maxwell instruction set
			{
				this->m_builder.AddInstruction(new SASS::Maxwell::IADD32IInstruction(
					temp_Lo, reg_Lo, addressImmediate, SASS::Maxwell::IADD32IInstruction::Flags::CC
				));

				if constexpr(B == PTX::Bits::Bits64)
				{
					this->m_builder.AddInstruction(new SASS::Maxwell::IADD32IInstruction(
						temp_Hi, reg_Hi, new SASS::I32Immediate(0x0), SASS::Maxwell::IADD32IInstruction::Flags::X
					));
				}
			},
			[&]() // Volta instruction set
			{
				if constexpr(B == PTX::Bits::Bits32)
				{
					this->m_builder.AddInstruction(new SASS::Volta::IADD3Instruction(
						temp_Lo, reg_Lo, addressImmediate, SASS::RZ
					));
				}
				else
				{
					auto CC = this->m_builder.AllocateTemporaryPredicate();

					this->m_builder.AddInstruction(new SASS::Volta::IADD3Instruction(
						temp_Lo, CC, reg_Lo, addressImmediate, SASS::RZ
					));
					this->m_builder.AddInstruction(new SASS::Volta::IADD3Instruction(
						temp_Hi, reg_Hi, SASS::RZ, SASS::RZ, CC, SASS::PT, SASS::Volta::IADD3Instruction::Flags::NOT_E
					));
				}
			});

			auto temp = new SASS::Register(temp_Lo->GetValue(), size);
			m_address = new SASS::Address(temp);
		}
		else
		{
			auto reg = new SASS::Register(reg_Lo->GetValue(), size);
			m_address = new SASS::Address(reg, addressOffset);
		}
	}
	else if constexpr(std::is_same<S, PTX::SharedSpace>::value)
	{
		// Inline offset has maximum size

		if (addressOffset >= (1 << 24) || !m_useOffset)
		{
			auto temp = this->m_builder.AllocateTemporaryRegister();
			auto addressImmediate = new SASS::I32Immediate(addressOffset);

			ArchitectureDispatch::DispatchInline(m_builder,
			[&]() // Maxwell instruction set
			{
				this->m_builder.AddInstruction(new SASS::Maxwell::IADD32IInstruction(temp, reg_Lo, addressImmediate));
			},
			[&]() // Volta instruction set
			{
				this->m_builder.AddInstruction(new SASS::Volta::IADD3Instruction(temp, reg_Lo, addressImmediate, SASS::RZ));
			});
			m_address = new SASS::Address(temp);
		}
		else
		{
			m_address = new SASS::Address(reg_Lo, addressOffset);
		}
	}
	else
	{
		Error(address, "unsupported space");
	}
}

}
}
