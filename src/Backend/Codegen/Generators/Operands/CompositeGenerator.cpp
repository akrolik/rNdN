#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"

#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

std::pair<SASS::Composite *, SASS::Composite *> CompositeGenerator::Generate(const PTX::Operand *operand)
{
	// Clear

	m_composite = nullptr;
	m_compositeHi = nullptr;

	// Generate composite

	operand->Accept(*this);
	if (m_composite == nullptr)
	{
		Error(operand, "unsupported kind");
	}
	return { m_composite, m_compositeHi };
}

void CompositeGenerator::Visit(const PTX::_Register *reg)
{
	reg->Dispatch(*this);
}

void CompositeGenerator::Visit(const PTX::_IndexedRegister *reg)
{
	reg->Dispatch(*this);
}

template<class T>
void CompositeGenerator::Visit(const PTX::Register<T> *reg)
{
	RegisterGenerator registerGenerator(this->m_builder);
	auto [regLo, regHi] = registerGenerator.Generate(reg);

	m_composite = regLo;
	m_compositeHi = regHi;
}

template<class T, class S, PTX::VectorSize V>
void CompositeGenerator::Visit(const PTX::IndexedRegister<T, S, V> *reg)
{
	RegisterGenerator registerGenerator(this->m_builder);
	auto [regLo, regHi] = registerGenerator.Generate(reg);

	m_composite = regLo;
	m_compositeHi = regHi;
}

void CompositeGenerator::Visit(const PTX::_Constant *constant)
{
	constant->Dispatch(*this);
}

void CompositeGenerator::Visit(const PTX::_Value *value)
{
	value->Dispatch(*this);
}

template<class T>
void CompositeGenerator::Visit(const PTX::Constant<T> *constant)
{
	if constexpr(std::is_same<T, PTX::UInt32Type>::value)
	{
		// Architecture property

		if (constant->GetName() == PTX::SpecialConstantName_WARP_SZ)
		{
			m_composite = new SASS::I32Immediate(32);
			return;
		}
	}

	Error(constant,  "constant not found");
}

template<class T>
void CompositeGenerator::Visit(const PTX::Value<T> *value)
{
	if (value->GetValue() == 0)
	{
		if (m_zeroRegister)
		{
			m_composite = SASS::RZ;
			if constexpr(T::TypeBits == PTX::Bits::Bits64)
			{
				m_compositeHi = SASS::RZ;
			}
		}
		else
		{
			m_composite = new SASS::I32Immediate(0);
			if constexpr(T::TypeBits == PTX::Bits::Bits64)
			{
				m_compositeHi = new SASS::I32Immediate(0);
			}
		}
	}
	else
	{
		//TODO: Decide which Value<T> types are loading using MOV, constant 0x2, and immediates

		if constexpr(PTX::is_int_type<T>::value)
		{
			if (m_immediateValue)
			{
				if (value->GetValue() < 0xffffff)
				{
					m_composite = new SASS::I32Immediate(value->GetValue());
					if constexpr(T::TypeBits == PTX::Bits::Bits64)
					{
						m_compositeHi = new SASS::I32Immediate(0x0);
					}
					return;
				}
			}
		}

		// Allocate space in constant memory

		auto offset = this->m_builder.AddConstantMemory(value->GetValue());

		// Create composite value (hi for 64-bit types)

		m_composite = new SASS::Constant(0x2, offset);
		if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			m_compositeHi = new SASS::Constant(0x2, offset + 0x4);
		}
	}
}

void CompositeGenerator::Visit(const PTX::_MemoryAddress *address)
{
	address->Dispatch(*this);
}

template<PTX::Bits B, class T, class S>
void CompositeGenerator::Visit(const PTX::MemoryAddress<B, T, S> *address)
{
	const auto& allocations = this->m_builder.GetParameterSpaceAllocation();

	// Verify parameter allocated

	const auto& name = address->GetVariable()->GetName();
	if (allocations->ContainsParameter(name))
	{
		auto offset = allocations->GetParameterOffset(name);
		m_composite = new SASS::Constant(0x0, offset);

		// Extended datatypes

		if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			m_compositeHi = new SASS::Constant(0x0, offset + 0x4);
		}
	}
	else
	{
		Error(address, "parameter not found");
	}
}

}
}
