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
		Error(operand, "unsupported kind");
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

		if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			// Extended datatypes
			
			if (range == 2)
			{
				m_registerHi = new SASS::Register(allocation + 1);
			}
		}
	}
	else
	{
		// Non-allocated registers are given a temporary

		auto [temp, tempHi] = this->m_builder.AllocateTemporaryRegisterPair<T::TypeBits>();

		m_register = temp;
		m_registerHi = tempHi;
	}
}

template<class T, class S, PTX::VectorSize V>
void RegisterGenerator::Visit(const PTX::IndexedRegister<T, S, V> *reg)
{
	Error(reg);
}

template<class T>
void RegisterGenerator::Visit(const PTX::SinkRegister<T> *reg)
{
	// Sink register given a temporary

	auto [temp, tempHi] = this->m_builder.AllocateTemporaryRegisterPair<T::TypeBits>();

	m_register = temp;
	m_registerHi = tempHi;
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
	auto [reg, regHi] = this->m_builder.AllocateTemporaryRegisterPair<T::TypeBits>();

	m_register = reg;
	m_registerHi = regHi;
		
	if constexpr(std::is_same<T, PTX::UInt32Type>::value)
	{
		// Architecture property

		if (constant->GetName() == PTX::SpecialConstantName_WARP_SZ)
		{
			this->m_builder.AddInstruction(new SASS::MOV32IInstruction(m_register, new SASS::I32Immediate(32)));
			return;
		}
	}

	Error(constant, "constant not found");
}

template<class T>
void RegisterGenerator::Visit(const PTX::Value<T> *value)
{
	if (value->GetValue() == 0)
	{
		m_register = SASS::RZ;
	}
	else
	{
		auto [reg, regHi] = this->m_builder.AllocateTemporaryRegisterPair<T::TypeBits>();

		m_register = reg;
		m_registerHi = regHi;
		
		if constexpr(PTX::is_int_type<T>::value && PTX::BitSize<T::TypeBits>::NumBits <= 32)
		{
			this->m_builder.AddInstruction(new SASS::MOV32IInstruction(m_register, new SASS::I32Immediate(value->GetValue())));
		}
		else
		{
			// Allocate space in constant memory

			auto offset = this->m_builder.AddConstantMemory(value->GetValue());

			// Load value from constant space into registers (hi for 64-bit types)

			auto constant = new SASS::Constant(0x2, offset);
			this->m_builder.AddInstruction(new SASS::MOVInstruction(m_register, constant));

			if constexpr(T::TypeBits == PTX::Bits::Bits64)
			{
				auto constantHi = new SASS::Constant(0x2, offset + 0x4);
				this->m_builder.AddInstruction(new SASS::MOVInstruction(m_registerHi, constantHi));
			}
		}
	}
}

}
}
