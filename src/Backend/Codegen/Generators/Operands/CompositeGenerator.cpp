#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"

#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

SASS::Composite *CompositeGenerator::Generate(const PTX::Operand *operand)
{
	// Clear

	m_composite = nullptr;
	m_compositeHi = nullptr;
	m_pair = false;

	// Generate composite

	operand->Accept(*this);
	if (m_composite == nullptr)
	{
		Error(operand, "unsupported kind");
	}
	return m_composite;
}

std::pair<SASS::Composite *, SASS::Composite *> CompositeGenerator::GeneratePair(const PTX::Operand *operand)
{
	// Clear

	m_composite = nullptr;
	m_compositeHi = nullptr;
	m_pair = true;

	// Generate composite

	operand->Accept(*this);
	if (m_composite == nullptr)
	{
		Error(operand, "unsupported kind");
	}
	return { m_composite, m_compositeHi };
}

bool CompositeGenerator::Visit(const PTX::_Register *reg)
{
	reg->Dispatch(*this);
	return false;
}

bool CompositeGenerator::Visit(const PTX::_IndexedRegister *reg)
{
	reg->Dispatch(*this);
	return false;
}

template<class T>
void CompositeGenerator::Visit(const PTX::Register<T> *reg)
{
	RegisterGenerator registerGenerator(this->m_builder);
	if (m_pair)
	{
		auto [regLo, regHi] = registerGenerator.GeneratePair(reg);

		m_composite = regLo;
		m_compositeHi = regHi;
	}
	else
	{
		m_composite = registerGenerator.Generate(reg);
	}
}

template<class T, class S, PTX::VectorSize V>
void CompositeGenerator::Visit(const PTX::IndexedRegister<T, S, V> *reg)
{
	RegisterGenerator registerGenerator(this->m_builder);
	if (m_pair)
	{
		auto [regLo, regHi] = registerGenerator.GeneratePair(reg);

		m_composite = regLo;
		m_compositeHi = regHi;
	}
	else
	{
		m_composite = registerGenerator.Generate(reg);
	}
}

bool CompositeGenerator::Visit(const PTX::_Constant *constant)
{
	constant->Dispatch(*this);
	return false;
}

bool CompositeGenerator::Visit(const PTX::_ParameterConstant *constant)
{
	constant->Dispatch(*this);
	return false;
}

bool CompositeGenerator::Visit(const PTX::_Value *value)
{
	value->Dispatch(*this);
	return false;
}

template<class T>
void CompositeGenerator::Visit(const PTX::Constant<T> *constant)
{
	if constexpr(std::is_same<T, PTX::UInt32Type>::value)
	{
		// Architecture property

		if (constant->GetName() == PTX::SpecialConstantName_WARP_SZ)
		{
			m_composite = new SASS::I32Immediate(SASS::WARP_SIZE);
			return;
		}
	}

	Error(constant,  "constant not found");
}

template<class T>
void CompositeGenerator::Visit(const PTX::ParameterConstant<T> *constant)
{
	const auto& allocations = this->m_builder.GetParameterSpaceAllocation();

	// Verify parameter allocated

	const auto& name = constant->GetName();
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
		Error(constant, "parameter not found");
	}
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
			m_composite = new SASS::I32Immediate(0x0);
			if (m_immediateSize <= 8)
			{
				m_composite = new SASS::I8Immediate(0x0);
			}
			else if (m_immediateSize <= 16)
			{
				m_composite = new SASS::I16Immediate(0x0);
			}
			else
			{
				m_composite = new SASS::I32Immediate(0x0);
				if constexpr(T::TypeBits == PTX::Bits::Bits64)
				{
					m_compositeHi = new SASS::I32Immediate(0x0);
				}
			}
		}
	}
	else
	{
		// If possible, store the value within the instruction itself

		if (m_immediateValue)
		{
			if constexpr(PTX::is_int_type<T>::value)
			{
				// Ensure the value can be represented in the required number of bits

				auto limit = (static_cast<std::uint64_t>(1) << (m_immediateSize + 1) - 1);
				if (auto val = value->GetValue(); val < limit)
				{
					if (m_immediateSize <= 8)
					{
						m_composite = new SASS::I8Immediate(val);
					}
					else if (m_immediateSize <= 16)
					{
						m_composite = new SASS::I16Immediate(val);
					}
					else
					{
						m_composite = new SASS::I32Immediate(val);
						if constexpr(T::TypeBits == PTX::Bits::Bits64)
						{
							m_compositeHi = new SASS::I32Immediate(0x0);
						}
					}
					return;
				}
			}
			else if constexpr(PTX::is_float_type<T>::value)
			{
				// For floating point, we keep the beginning bits (opposite to integer)

				auto shift = PTX::BitSize<T::TypeBits>::NumBits - m_immediateSize;
				auto limit = (1 << (m_immediateSize + 1) - 1) << shift;

				auto val = value->GetValue();
				auto bitVal = reinterpret_cast<std::uint64_t&>(val);

				if (bitVal & limit == bitVal)
				{
					if constexpr(T::TypeBits == PTX::Bits::Bits32)
					{
						m_composite = new SASS::F32Immediate(val);
					}
					else if constexpr(T::TypeBits == PTX::Bits::Bits64)
					{
						m_composite = new SASS::F64Immediate(val);
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

bool CompositeGenerator::Visit(const PTX::_MemoryAddress *address)
{
	address->Dispatch(*this);
	return false;
}

template<PTX::Bits B, class T, class S>
void CompositeGenerator::Visit(const PTX::MemoryAddress<B, T, S> *address)
{
	if constexpr(std::is_same<S, PTX::ParameterSpace>::value)
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
	else
	{
		Error(address, "unsupported space");
	}
}

}
}
