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
		Error("address for operand '" + PTX::PrettyPrinter::PrettyString(operand) + "'");
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
		const auto& globalAllocations = this->m_builder.GetGlobalSpaceAllocation();
		if (globalAllocations->ContainsGlobalMemory(name))
		{
			// Global addresses are initially zero and relocated at runtime

			auto temp0 = this->m_builder.AllocateTemporaryRegister();
			auto inst0 = new SASS::MOV32IInstruction(temp0, new SASS::I32Immediate(0x0));

			this->m_builder.AddInstruction(inst0);
			this->m_builder.AddRelocation(inst0, name, SASS::Relocation::Kind::ABS32_LO_20);

			// Extended addresses (64-bit)

			if constexpr(B == PTX::Bits::Bits64)
			{
				auto temp1 = this->m_builder.AllocateTemporaryRegister();
				auto inst1 = new SASS::MOV32IInstruction(temp1, new SASS::I32Immediate(0x0));

				this->m_builder.AddInstruction(inst1);
				this->m_builder.AddRelocation(inst1, name, SASS::Relocation::Kind::ABS32_HI_20);
			}

			// Form the address with the offset

			m_address = new SASS::Address(temp0, address->GetOffset());
		}
	}
	else if constexpr(std::is_same<S, PTX::SharedSpace>::value)
	{
		auto variableAddress = 0x0;
		auto variableFound = false;

		// Check if shared variable locally allocated

		const auto& localAllocations = this->m_builder.GetLocalSpaceAllocation();
		if (localAllocations->ContainsSharedMemory(name))
		{
			variableAddress = localAllocations->GetSharedMemoryOffset(name);
		}
		else
		{
			// Check if shared variable globally (module) allocated

			const auto& globalAllocations = this->m_builder.GetGlobalSpaceAllocation();
			if (globalAllocations->ContainsSharedMemory(name))
			{
				variableAddress = globalAllocations->GetSharedMemoryOffset(name);
			}
			else if (globalAllocations->ContainsDynamicSharedMemory(name))
			{
				// Dynamic shared memory is module allocated but offset by local space

				variableAddress = localAllocations->GetDynamicSharedMemoryOffset();
			}
		}

		if (variableFound)
		{
			// For shared variables, use an absolute address computed as the variable location + offset

			auto offset = address->GetOffset();
			m_address = new SASS::Address(SASS::RZ, variableAddress + offset);
		}
	}
}

template<PTX::Bits B, class T, class S>
void AddressGenerator::Visit(const PTX::RegisterAddress<B, T, S> *address)
{
	// Generate register for address

	RegisterGenerator registerGenerator(this->m_builder);
	auto [reg, regHi] = registerGenerator.Generate(address->GetRegister());

	// Construct address with offset

	m_address = new SASS::Address(reg, address->GetOffset());
}

}
}
