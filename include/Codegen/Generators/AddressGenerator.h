#pragma once

#include <cmath>

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/IndexGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class AddressGenerator : public Generator
{
public:
	using Generator::Generator;

	using IndexKind = IndexGenerator::Kind;

	template<class T, class S>
	PTX::Address<B, T, S> *Generate(const PTX::Variable<T, S> *variable, IndexKind indexKind, unsigned int offset = 0)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Get the base address from the variable

		auto base = new PTX::PointerRegisterAdapter<B, T, S>(resources->template AllocateTemporary<PTX::UIntType<B>>());
		auto baseAddress = new PTX::MemoryAddress<B, T, PTX::SharedSpace>(variable, offset);

		// Take the address of the variable using the move instruction

		this->m_builder.AddStatement(new PTX::MoveAddressInstruction<B, T, S>(base, baseAddress));

		return GenerateBase(base, indexKind);
	}

	template<class T, class S>
	PTX::Address<B, T, S> *GenerateParameter(const PTX::ParameterVariable<PTX::PointerType<B, T>> *variable, IndexKind indexKind)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Get the base address of the variable in generic space

		auto genericBase = new PTX::PointerRegisterAdapter<B, T>(resources->template AllocateTemporary<PTX::UIntType<B>>());
		auto spaceBase = new PTX::PointerRegisterAdapter<B, T, S>(resources->template AllocateTemporary<PTX::UIntType<B>>());

		auto baseAddress = new PTX::MemoryAddress<B, PTX::PointerType<B, T>, PTX::ParameterSpace>(variable);

		// Load the generic address of the underlying variable from the parameter space,
		// and convert it to the underlying space

		this->m_builder.AddStatement(new PTX::LoadInstruction<B, PTX::PointerType<B, T>, PTX::ParameterSpace>(genericBase, baseAddress));
		this->m_builder.AddStatement(new PTX::ConvertToAddressInstruction<B, T, S>(spaceBase, new PTX::RegisterAddress<B, T>(genericBase)));

		return GenerateBase(spaceBase, indexKind);
	}

private:
	template<class T, class S>
	PTX::Address<B, T, S> *GenerateBase(const PTX::PointerRegisterAdapter<B, T, S> *base, IndexKind indexKind)
	{
		auto resources = this->m_builder.GetLocalResources();

		auto offset = resources->template AllocateTemporary<PTX::UIntType<B>>();
		auto address = resources->template AllocateTemporary<PTX::UIntType<B>>();

		IndexGenerator indexGen(this->m_builder);
		auto index = indexGen.GenerateIndex(indexKind);

		// Compute offset from the base address using the thread id and the data size

		if constexpr(B == PTX::Bits::Bits32)
		{
			// In a 32-bit system, the offset is computed by using a left shift of the thread id by
			// the data size. The registers are both adapted to bit types since it is required
			// for the instruction

			this->m_builder.AddStatement(new PTX::ShiftLeftInstruction<PTX::Bit32Type>(
				new PTX::Bit32RegisterAdapter<PTX::UIntType>(offset),
				new PTX::Bit32Adapter<PTX::UIntType>(index),
				new PTX::UInt32Value(std::log2(PTX::BitSize<T::TypeBits>::NumBytes))
			));
		}
		else
		{
			// In a 64-bit system, the offset is computed using a wide multiplication of the thread
			// id and the data size. A wide multiplication extends the result past the 32-bit
			// size of both operands

			this->m_builder.AddStatement(new PTX::MultiplyWideInstruction<PTX::UInt32Type>(offset, index, new PTX::UInt32Value(PTX::BitSize<T::TypeBits>::NumBytes)));
		}

		// Sum the base and the offset to create the full address for the thread and store the value in a register

		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UIntType<B>>(address, base->GetVariable(), offset));

		// Create an address from the resulting sum

		return new PTX::RegisterAddress<B, T, S>(new PTX::PointerRegisterAdapter<B, T, S>(address));
	}
};

}
