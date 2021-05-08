#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "PTX/Utils/PrettyPrinter.h"

namespace Backend {
namespace Codegen {

SASS::Register *RegisterGenerator::Generate(const PTX::Operand *operand)
{
	// Clear

	m_register = nullptr;
	m_registerHi = nullptr;
	m_pair = false;

	// Generate register

	operand->Accept(*this);
	if (m_register == nullptr)
	{
		Error(operand, "unsupported kind");
	}
	return m_register;
}

std::pair<SASS::Register *, SASS::Register *> RegisterGenerator::GeneratePair(const PTX::Operand *operand)
{
	// Clear

	m_register = nullptr;
	m_registerHi = nullptr;
	m_pair = true;

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
		if (m_pair)
		{
			m_register = new SASS::Register(allocation);
			if constexpr(T::TypeBits == PTX::Bits::Bits64)
			{
				m_registerHi = new SASS::Register(allocation + 1);
			}
		}
		else
		{
			m_register = new SASS::Register(allocation, range);
		}
	}
	else
	{
		// Non-allocated registers are given a temporary

		if (m_pair)
		{
			auto [temp, tempHi] = this->m_builder.AllocateTemporaryRegisterPair<T::TypeBits>();

			m_register = temp;
			m_registerHi = tempHi;
		}
		else
		{
			m_register = this->m_builder.AllocateTemporaryRegister<T::TypeBits>();
		}
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

	if (m_pair)
	{
		auto [temp, tempHi] = this->m_builder.AllocateTemporaryRegisterPair<T::TypeBits>();

		m_register = temp;
		m_registerHi = tempHi;
	}
	else
	{
		m_register = this->m_builder.AllocateTemporaryRegister<T::TypeBits>();
	}
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
	if constexpr(std::is_same<T, PTX::UInt32Type>::value)
	{
		// Architecture property

		if (constant->GetName() == PTX::SpecialConstantName_WARP_SZ)
		{
			m_register = this->m_builder.AllocateTemporaryRegister<T::TypeBits>();
		
			this->m_builder.AddInstruction(new SASS::MOV32IInstruction(m_register, new SASS::I32Immediate(SASS::WARP_SIZE)));
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
		if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			m_registerHi = SASS::RZ;
		}
	}
	else
	{
		if constexpr(PTX::is_int_type<T>::value && PTX::BitSize<T::TypeBits>::NumBits <= 32)
		{
			m_register = this->m_builder.AllocateTemporaryRegister<T::TypeBits>();

			this->m_builder.AddInstruction(new SASS::MOV32IInstruction(m_register, new SASS::I32Immediate(value->GetValue())));
		}
		else
		{
			auto [reg, regHi] = this->m_builder.AllocateTemporaryRegisterPair<T::TypeBits>();

			// Allocate space in constant memory

			auto offset = this->m_builder.AddConstantMemory(value->GetValue());

			// Load value from constant space into registers (hi for 64-bit types)

			auto constant = new SASS::Constant(0x2, offset);
			this->m_builder.AddInstruction(new SASS::MOVInstruction(reg, constant));

			if constexpr(T::TypeBits == PTX::Bits::Bits64)
			{
				auto constantHi = new SASS::Constant(0x2, offset + 0x4);
				this->m_builder.AddInstruction(new SASS::MOVInstruction(regHi, constantHi));
			}

			if (m_pair)
			{
				m_register = reg;
				m_registerHi = regHi;
			}
			else
			{
				auto size = Utils::Math::DivUp(PTX::BitSize<T::TypeBits>::NumBits, 32);
				m_register = new SASS::Register(reg->GetValue(), size);
			}
		}
	}
}

}
}
