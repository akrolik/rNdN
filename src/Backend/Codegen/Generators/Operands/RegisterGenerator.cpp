#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "PTX/Utils/PrettyPrinter.h"

namespace Backend {
namespace Codegen {

SASS::Register *RegisterGenerator::Generate(const PTX::Operand *operand)
{
	operand->Accept(*this);
	if (m_register == nullptr)
	{
		Error("register for operand '" + PTX::PrettyPrinter::PrettyString(operand) + "'");
	}
	return m_register;
}

void RegisterGenerator::Visit(const PTX::_Register *reg)
{
	reg->Dispatch(*this);
}

void RegisterGenerator::Visit(const PTX::_IndexedRegister *reg)
{
	reg->Dispatch(*this);
}

template<class T>
void RegisterGenerator::Visit(const PTX::Register<T> *reg)
{
	const auto& allocations = this->m_builder.GetRegisterAllocation();

	// Verify register allocated

	const auto& name = reg->GetName();
	if (allocations->ContainsRegister(name))
	{
		const auto& [allocation, _] = allocations->GetRegister(name);
		m_register = new SASS::Register(allocation);
	}
}

template<class T, class S, PTX::VectorSize V>
void RegisterGenerator::Visit(const PTX::IndexedRegister<T, S, V> *reg)
{
	//TODO: Indexed registers
}

}
}
