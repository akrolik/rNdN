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
	if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
	{
		// Global addresses are initially zero and relocated at runtime
		//TODO: Rellocatable ELF globals

		auto temp0 = this->m_builder.AllocateTemporaryRegister();
		this->m_builder.AddInstruction(new SASS::MOV32IInstruction(temp0, new SASS::I32Immediate(0x0)));

		// Extended addresses (64-bit)

		if constexpr(B == PTX::Bits::Bits64)
		{
			auto temp1 = this->m_builder.AllocateTemporaryRegister();
			this->m_builder.AddInstruction(new SASS::MOV32IInstruction(temp1, new SASS::I32Immediate(0x0)));
		}

		// Form the address with the offset

		m_address = new SASS::Address(temp0, address->GetOffset());
	}
	else if constexpr(std::is_same<S, PTX::SharedSpace>::value)
	{
		const auto& allocations = this->m_builder.GetSpaceAllocation();

		// Verify shared variable allocated

		const auto& name = address->GetVariable()->GetName();
		if (allocations->ContainsSharedVariable(name))
		{
			// For shared variables, use an absolute address computed as the variable location + offset

			auto variable = allocations->GetSharedVariableOffset(name);
			auto offset = address->GetOffset();

			m_address = new SASS::Address(SASS::RZ, variable + offset);
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
