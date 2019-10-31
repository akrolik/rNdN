#pragma once

#include <cmath>

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class AddressGenerator : public Generator
{
public:
	using Generator::Generator;

	template<class T>
	PTX::Address<B, T, PTX::GlobalSpace> *GenerateAddress(const PTX::ParameterVariable<PTX::PointerType<B, T>> *variable, const PTX::TypedOperand<PTX::UInt32Type> *index = nullptr)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Get the special address register

		auto dataAddressName = NameUtils::DataAddressName(variable);
		auto addressRegister = resources->GetRegister<PTX::UIntType<B>>(dataAddressName);

		// Generate the address for the correct index

		return GenerateAddress<T, PTX::GlobalSpace>(addressRegister, index);
	}

	template<class T>
	PTX::Address<B, T, PTX::GlobalSpace> *GenerateAddress(const PTX::ParameterVariable<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>> *variable, const PTX::TypedOperand<PTX::UInt32Type> *index = nullptr)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Get the special address register

		auto dataAddressName = NameUtils::DataAddressName(variable);
		auto addressRegister = resources->GetRegister<PTX::UIntType<B>>(dataAddressName);

		// Generate the address for the correct index

		return GenerateAddress<T, PTX::GlobalSpace>(addressRegister, index);
	}

	template<class T, class S>
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

		return GenerateAddress<T, S>(base, index);
	}

	template<class T>
	void LoadParameterAddress(const PTX::ParameterVariable<PTX::PointerType<B, T>> *parameter)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Get the base address of the variable in generic space

		auto baseAddress = new PTX::MemoryAddress<B, PTX::PointerType<B, T>, PTX::ParameterSpace>(parameter);
		auto genericBase = new PTX::PointerRegisterAdapter<B, T>(resources->template AllocateTemporary<PTX::UIntType<B>>());

		this->m_builder.AddStatement(new PTX::LoadInstruction<B, PTX::PointerType<B, T>, PTX::ParameterSpace>(genericBase, baseAddress));

		// Convert the generic address of the underlying variable to the global space

		auto name = NameUtils::DataAddressName(parameter);
		auto globalBase = new PTX::PointerRegisterAdapter<B, T, PTX::GlobalSpace>(resources->template AllocateRegister<PTX::UIntType<B>>(name));
		auto genericAddress = new PTX::RegisterAddress<B, T>(genericBase);

		this->m_builder.AddStatement(new PTX::ConvertToAddressInstruction<B, T, PTX::GlobalSpace>(globalBase, genericAddress));
	}

	template<class T>
	void LoadParameterAddress(const PTX::ParameterVariable<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>> *parameter, const PTX::TypedOperand<PTX::UInt32Type> *index)
	{
		using DataType = PTX::PointerType<B, T, PTX::GlobalSpace>;

		auto resources = this->m_builder.GetLocalResources();

		// Get the base address of the variable in generic space

		auto baseAddress = new PTX::MemoryAddress<B, PTX::PointerType<B, DataType>, PTX::ParameterSpace>(parameter);
		auto genericBase = new PTX::PointerRegisterAdapter<B, DataType>(resources->template AllocateTemporary<PTX::UIntType<B>>());

		this->m_builder.AddStatement(new PTX::LoadInstruction<B, PTX::PointerType<B, DataType>, PTX::ParameterSpace>(genericBase, baseAddress));

		// Convert the generic address of the underlying variable to the global space

		auto genericAddress = new PTX::RegisterAddress<B, DataType>(genericBase);
		auto globalBase = new PTX::PointerRegisterAdapter<B, DataType, PTX::GlobalSpace>(resources->template AllocateTemporary<PTX::UIntType<B>>());

		this->m_builder.AddStatement(new PTX::ConvertToAddressInstruction<B, DataType, PTX::GlobalSpace>(globalBase, genericAddress));

		// Get the address of the value in the indirection structure (by the bitsize of the address)

		auto globalIndexed = new PTX::PointerRegisterAdapter<B, DataType, PTX::GlobalSpace>(resources->template AllocateTemporary<PTX::UIntType<B>>());
		auto offset = GenerateAddressOffset<B>(index);

		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UIntType<B>>(globalIndexed->GetVariable(), globalBase->GetVariable(), offset));

		// Load the address of the data

		auto name = NameUtils::DataAddressName(parameter);
		auto dataPointer = new PTX::PointerRegisterAdapter<B, T, PTX::GlobalSpace>(resources->template AllocateRegister<PTX::UIntType<B>>(name));
		auto indexedAddress = new PTX::RegisterAddress<B, DataType, PTX::GlobalSpace>(globalIndexed);

		this->m_builder.AddStatement(new PTX::LoadInstruction<B, DataType, PTX::GlobalSpace>(dataPointer, indexedAddress));
	}

	template<class T, class S>
	PTX::RegisterAddress<B, T, S> *GenerateAddress(const PTX::Register<PTX::UIntType<B>> *base, const PTX::TypedOperand<PTX::UInt32Type> *index = nullptr)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Sum the base and the offset to create the full address for the thread and store the value in a register

		if (index == nullptr)
		{
			return new PTX::RegisterAddress(new PTX::PointerRegisterAdapter<B, T, S>(base));
		}
		else
		{
			auto address = resources->template AllocateTemporary<PTX::UIntType<B>>();
			auto offset = GenerateAddressOffset<T::TypeBits>(index);

			this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UIntType<B>>(address, base, offset));

			return new PTX::RegisterAddress(new PTX::PointerRegisterAdapter<B, T, S>(address));
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
};

}
