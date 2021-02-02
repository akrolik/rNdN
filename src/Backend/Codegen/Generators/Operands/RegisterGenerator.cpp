#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "PTX/Utils/PrettyPrinter.h"

namespace Backend {
namespace Codegen {

std::pair<SASS::Register *, SASS::Register *> RegisterGenerator::Generate(const PTX::Operand *operand)
{
	// Clear

	m_register = nullptr;
	m_registerHi = nullptr;

	// Generate register

	operand->Accept(*this);
	if (m_register == nullptr)
	{
		//TODO: Temporarily use the first scratch register for unallocated variables
		// Error("register for operand '" + PTX::PrettyPrinter::PrettyString(operand) + "'");
		return { this->m_builder.AllocateTemporaryRegister(), nullptr };
	}
	return { m_register, m_registerHi };
}

void RegisterGenerator::Visit(const PTX::_Register *reg)
{
	reg->Dispatch(*this);
}

void RegisterGenerator::Visit(const PTX::_IndexedRegister *reg)
{
	reg->Dispatch(*this);
}

void RegisterGenerator::Visit(const PTX::_SinkRegister *reg)
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
		const auto& [allocation, range] = allocations->GetRegister(name);
		m_register = new SASS::Register(allocation);

		// Extended datatypes
		
		if (range == 2)
		{
			m_registerHi = new SASS::Register(allocation + 1);
		}
	}
}

template<class T, class S, PTX::VectorSize V>
void RegisterGenerator::Visit(const PTX::IndexedRegister<T, S, V> *reg)
{
	//TODO: Indexed registers
}

template<class T>
void RegisterGenerator::Visit(const PTX::SinkRegister<T> *reg)
{
	//TODO: Sink register
}

void RegisterGenerator::Visit(const PTX::_Constant *constant)
{
	constant->Dispatch(*this);
}

void RegisterGenerator::Visit(const PTX::_Value *value)
{
	value->Dispatch(*this);
}

template<class T>
void RegisterGenerator::Visit(const PTX::Constant<T> *constant)
{
	//TODO: Constant
}

template<class T>
void RegisterGenerator::Visit(const PTX::Value<T> *value)
{
	if (value->GetValue() == 0)
	{
		m_register = SASS::RZ;
	}
	// if constexpr(PTX::TypeBits == PTX::Bits::Bits64)
	// {
	// 	//TODO: Move other values into register
	// }
}

}
}
