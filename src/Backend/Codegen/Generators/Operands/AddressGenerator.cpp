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
	//TODO: Memory address
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
