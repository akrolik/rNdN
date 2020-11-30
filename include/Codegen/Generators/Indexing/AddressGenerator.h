#pragma once

#include <cmath>

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"

#include "PTX/Tree/Tree.h"

namespace Codegen {

template<PTX::Bits B, class T>
class AddressGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "AddressGenerator"; }

	PTX::Address<B, T, PTX::GlobalSpace> *GenerateAddress(const PTX::ParameterVariable<PTX::PointerType<B, T>> *variable, const PTX::TypedOperand<PTX::UInt32Type> *index = nullptr, int offset = 0)
	{
		return GenerateAddress(NameUtils::DataAddressName(variable), index, offset);
	}

	PTX::Address<B, T, PTX::GlobalSpace> *GenerateAddress(const PTX::ParameterVariable<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>> *variable, const PTX::TypedOperand<PTX::UInt32Type> *index = nullptr, int offset = 0)
	{
		return GenerateAddress(NameUtils::DataCellAddressName(variable), index, offset);
	}

	PTX::Address<B, T, PTX::GlobalSpace> *GenerateAddress(const PTX::ParameterVariable<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>> *variable, unsigned int cellIndex, const PTX::TypedOperand<PTX::UInt32Type> *index = nullptr, int offset = 0)
	{
		return GenerateAddress(NameUtils::DataCellAddressName(variable, cellIndex), index, offset);
	}

	template<class S>
	PTX::Address<B, T, S> *GenerateAddress(const PTX::Variable<T, S> *variable, const PTX::TypedOperand<PTX::UInt32Type> *index = nullptr, int offset = 0)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Get the base address from the variable

		auto base = resources->template AllocateTemporary<PTX::UIntType<B>>();
		auto basePointer = new PTX::PointerRegisterAdapter<B, T, S>(base);
		auto baseAddress = new PTX::MemoryAddress<B, T, S>(variable, offset);
	
		// If there is no index, we can use the address directly

		if (index == nullptr)
		{
			return baseAddress;
		}

		// Take the address of the variable using the move instruction

		this->m_builder.AddStatement(new PTX::MoveAddressInstruction<B, T, S>(basePointer, baseAddress));

		// Create an address from the register

		return GenerateAddress<S>(base, index);
	}

	template<class S>
	PTX::RegisterAddress<B, T, S> *GenerateAddress(const PTX::Register<PTX::UIntType<B>> *base, const PTX::TypedOperand<PTX::UInt32Type> *index = nullptr, int offset = 0)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Sum the base and the offset to create the full address for the thread and store the value in a register

		if (index == nullptr)
		{
			return new PTX::RegisterAddress<B, T, S>(new PTX::PointerRegisterAdapter<B, T, S>(base), offset);
		}
		else
		{
			auto address = resources->template AllocateTemporary<PTX::UIntType<B>>();
			auto indexOffset = GenerateAddressOffset<T::TypeBits>(index);

			this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UIntType<B>>(address, base, indexOffset));

			return new PTX::RegisterAddress<B, T, S>(new PTX::PointerRegisterAdapter<B, T, S>(address), offset);
		}
	}

	template<PTX::Bits OffsetBits>
	const PTX::TypedOperand<PTX::UIntType<B>> *GenerateAddressOffset(const PTX::TypedOperand<PTX::UInt32Type> *index)
	{
		// Compute offset from the base address using the thread id and the data size

		auto resources = this->m_builder.GetLocalResources();
		auto offset = resources->template AllocateTemporary<PTX::UIntType<B>>();

		if constexpr(B == PTX::Bits::Bits32)
		{
			// In a 32-bit system, the offset is computed by using a left shift of the thread id by
			// the data size. The registers are both adapted to bit types since it is required
			// for the instruction

			this->m_builder.AddStatement(new PTX::ShiftLeftInstruction<PTX::Bit32Type>(
				new PTX::Bit32RegisterAdapter<PTX::UIntType>(offset),
				new PTX::Bit32Adapter<PTX::UIntType>(index),
				new PTX::UInt32Value(std::log2(PTX::BitSize<OffsetBits>::NumBytes))
			));
		}
		else
		{
			// In a 64-bit system, the offset is computed using a wide multiplication of the thread
			// id and the data size. A wide multiplication extends the result past the 32-bit
			// size of both operands

			this->m_builder.AddStatement(new PTX::MultiplyWideInstruction<PTX::UInt32Type>(offset, index, new PTX::UInt32Value(PTX::BitSize<OffsetBits>::NumBytes)));
		}

                return offset;
	}

private:
	PTX::Address<B, T, PTX::GlobalSpace> *GenerateAddress(const std::string& dataAddressName, const PTX::TypedOperand<PTX::UInt32Type> *index = nullptr, int offset = 0)
	{
		// Get the special address register

		auto resources = this->m_builder.GetLocalResources();
		auto addressRegister = resources->GetRegister<PTX::UIntType<B>>(dataAddressName);

		// Generate the address for the correct index

		return GenerateAddress<PTX::GlobalSpace>(addressRegister, index, offset);
	}

};

}
