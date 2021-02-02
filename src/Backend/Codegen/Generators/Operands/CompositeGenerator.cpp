#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"

#include "PTX/Utils/PrettyPrinter.h"

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
		Error("composite for operand '" + PTX::PrettyPrinter::PrettyString(operand) + "'");
	}
	return { m_composite, m_compositeHi };
}

void CompositeGenerator::Visit(const PTX::_Register *reg)
{
	reg->Dispatch(*this);
}

template<class T>
void CompositeGenerator::Visit(const PTX::Register<T> *reg)
{
	const auto& allocations = this->m_builder.GetRegisterAllocation();

	// Verify register allocated

	const auto& name = reg->GetName();
	if (allocations->ContainsRegister(name))
	{
		const auto& [allocation, range] = allocations->GetRegister(name);
		m_composite = new SASS::Register(allocation);

		// Extended datatypes

		if (range == 2)
		{
			m_compositeHi = new SASS::Register(allocation + 1);
		}
	}
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

		if (constant->GetName() == "WARP_SZ")
		{
			m_composite = new SASS::I32Immediate(32);
		}
	}
}

template<class T>
void CompositeGenerator::Visit(const PTX::Value<T> *value)
{
	if (value->GetValue() == 0)
	{
		m_composite = SASS::RZ;
		if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			m_compositeHi = SASS::RZ;
		}
	}
	else
	{
		//TODO: Decide which Value<T> types are loading using MOV, constant 0x2, and immediates

		if constexpr(PTX::is_int_type<T>::value && PTX::BitSize<T::TypeBits>::NumBits < 32)
		{
			m_composite = new SASS::I32Immediate(value->GetValue());
		}
		else
		{
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
}

void CompositeGenerator::Visit(const PTX::_MemoryAddress *address)
{
	address->Dispatch(*this);
}

template<PTX::Bits B, class T, class S>
void CompositeGenerator::Visit(const PTX::MemoryAddress<B, T, S> *address)
{
	const auto& name = address->GetVariable()->GetName();
	auto offset = this->m_builder.GetParameter(name);

	m_composite = new SASS::Constant(0x0, offset);

	// Extended datatypes

	if constexpr(T::TypeBits == PTX::Bits::Bits64)
	{
		m_compositeHi = new SASS::Constant(0x0, offset + 0x4);
	}
}

}
}
