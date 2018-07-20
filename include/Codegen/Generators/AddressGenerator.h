#pragma once

#include <cmath>

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"

#include "HorseIR/Tree/Statements/ReturnStatement.h"

#include "PTX/StateSpace.h"
#include "PTX/Type.h"
#include "PTX/Declarations/SpecialRegisterDeclarations.h"
#include "PTX/Functions/Function.h"
#include "PTX/Instructions/Arithmetic/AddInstruction.h"
#include "PTX/Instructions/Arithmetic/MultiplyWideInstruction.h"
#include "PTX/Instructions/Data/ConvertAddressInstruction.h"
#include "PTX/Instructions/Data/LoadInstruction.h"
#include "PTX/Instructions/Data/MoveAddressInstruction.h"
#include "PTX/Instructions/Data/MoveInstruction.h"
#include "PTX/Instructions/ControlFlow/ReturnInstruction.h"
#include "PTX/Instructions/Shift/ShiftLeftInstruction.h"
#include "PTX/Operands/Adapters/PointerAdapter.h"
#include "PTX/Operands/Adapters/BitAdapter.h"
#include "PTX/Operands/Address/MemoryAddress.h"
#include "PTX/Operands/Address/RegisterAddress.h"
#include "PTX/Operands/Variables/AddressableVariable.h"
#include "PTX/Operands/Variables/IndexedRegister.h"
#include "PTX/Operands/Variables/Variable.h"
#include "PTX/Operands/Value.h"

namespace Codegen {

template<PTX::Bits B>
class AddressGenerator : public Generator
{
public:
	using Generator::Generator;

	template<class T, class S>
	PTX::Address<B, T, S> *Generate(const PTX::Variable<T, S> *variable)
	{
		// Get the base address from the variable

		auto base = new PTX::PointerRegisterAdapter<B, T, S>(this->m_builder->template AllocateTemporary<PTX::UIntType<B>>());
		auto baseAddress = new PTX::MemoryAddress<B, T, PTX::SharedSpace>(variable);

		// Take the address of the variable using the move instruction

		this->m_builder->AddStatement(new PTX::MoveAddressInstruction<B, T, S>(base, baseAddress));

		return GenerateBase(base);
	}

	template<class T, class S>
	PTX::Address<B, T, S> *GenerateParameter(const PTX::ParameterVariable<PTX::PointerType<B, T>> *variable)
	{
		// Get the base address of the variable in generic space

		auto genericBase = new PTX::PointerRegisterAdapter<B, T>(this->m_builder->template AllocateTemporary<PTX::UIntType<B>>());
		auto spaceBase = new PTX::PointerRegisterAdapter<B, T, S>(this->m_builder->template AllocateTemporary<PTX::UIntType<B>>());

		auto baseAddress = new PTX::MemoryAddress<B, PTX::PointerType<B, T>, PTX::ParameterSpace>(variable);

		// Load the generic address of the underlying variable from the parameter space,
		// and convert it to the underlying space

		this->m_builder->AddStatement(new PTX::LoadInstruction<B, PTX::PointerType<B, T>, PTX::ParameterSpace>(genericBase, baseAddress));
		this->m_builder->AddStatement(new PTX::ConvertToAddressInstruction<B, T, S>(spaceBase, new PTX::RegisterAddress<B, T>(genericBase)));

		return GenerateBase(spaceBase);
	}

private:
	template<class T, class S>
	PTX::Address<B, T, S> *GenerateBase(const PTX::PointerRegisterAdapter<B, T, S> *base)
	{
		auto srtidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_tid->GetVariable("%tid"), PTX::VectorElement::X);
		auto tidx = this->m_builder->template AllocateTemporary<PTX::UInt32Type>("tidx");

		auto offset = this->m_builder->template AllocateTemporary<PTX::UIntType<B>>();
		auto address = this->m_builder->template AllocateTemporary<PTX::UIntType<B>>();

		// We cannot operate directly on special registers, so they must first be copied
		// to a use defined register
		
		this->m_builder->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(tidx, srtidx));

		// Compute offset from the base address using the thread id and the data size

		if constexpr(B == PTX::Bits::Bits32)
		{
			// In a 32-bit system, the offset is computed by using a left shift of the thread id by
			// the data size. The registers are both adapted to bit types since it is required
			// for the instruction

			this->m_builder->AddStatement(new PTX::ShiftLeftInstruction<PTX::Bit32Type>(
				new PTX::Bit32RegisterAdapter<PTX::UIntType>(offset),
				new PTX::Bit32RegisterAdapter<PTX::UIntType>(tidx),
				new PTX::UInt32Value(std::log2(PTX::BitSize<T::TypeBits>::NumBytes))
			));
		}
		else
		{
			// In a 64-bit system, the offset is computed using a wide multiplication of the thread
			// id and the data size. A wide multiplication extends the result past the 32-bit
			// size of both operands

			this->m_builder->AddStatement(new PTX::MultiplyWideInstruction<PTX::UInt32Type>(offset, tidx, new PTX::UInt32Value(PTX::BitSize<T::TypeBits>::NumBytes)));
		}

		// Sum the base and the offset to create the full address for the thread and store the value in a register

		this->m_builder->AddStatement(new PTX::AddInstruction<PTX::UIntType<B>>(address, base->GetVariable(), offset));

		// Create an address from the resulting sum

		return new PTX::RegisterAddress<B, T, S>(new PTX::PointerRegisterAdapter<B, T, S>(address));
	}
};

}
