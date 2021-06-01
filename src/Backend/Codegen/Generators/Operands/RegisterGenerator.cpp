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

bool RegisterGenerator::Visit(const PTX::_Register *reg)
{
	reg->Dispatch(*this);
	return false;
}

bool RegisterGenerator::Visit(const PTX::_IndexedRegister *reg)
{
	reg->Dispatch(*this);
	return false;
}

bool RegisterGenerator::Visit(const PTX::_SinkRegister *reg)
{
	reg->Dispatch(*this);
	return false;
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
		// Non-allocated registers are given system zero register

		m_register = SASS::RZ;
		if (m_pair)
		{
			m_registerHi = SASS::RZ;
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
	// Sink register given a system zero register

	m_register = SASS::RZ;
	if (m_pair)
	{
		m_registerHi = SASS::RZ;
	}
}

bool RegisterGenerator::Visit(const PTX::_Constant *constant)
{
	constant->Dispatch(*this);
	return false;
}

bool RegisterGenerator::Visit(const PTX::_ParameterConstant *constant)
{
	constant->Dispatch(*this);
	return false;
}

bool RegisterGenerator::Visit(const PTX::_Value *value)
{
	value->Dispatch(*this);
	return false;
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
void RegisterGenerator::Visit(const PTX::ParameterConstant<T> *constant)
{
	const auto& allocations = this->m_builder.GetParameterSpaceAllocation();

	// Verify parameter allocated

	const auto& name = constant->GetName();
	if (allocations->ContainsParameter(name))
	{
		auto [reg, regHi] = this->m_builder.AllocateTemporaryRegisterPair<T::TypeBits>();

		// Get offset in parameter constant

		auto offset = allocations->GetParameterOffset(name);

		// Load value from constant space into registers (hi for 64-bit types)

		auto constant = new SASS::Constant(0x0, offset);
		this->m_builder.AddInstruction(new SASS::MOVInstruction(reg, constant));
		m_register = reg;

		if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			auto constantHi = new SASS::Constant(0x0, offset + 0x4);
			this->m_builder.AddInstruction(new SASS::MOVInstruction(regHi, constantHi));
			m_registerHi = regHi;
		}
	}
	else
	{
		Error(constant, "parameter not found");
	}
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
