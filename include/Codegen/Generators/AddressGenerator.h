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
#include "PTX/Statements/BlockStatement.h"

namespace Codegen {

template<PTX::Bits B>
class AddressGenerator : public Generator
{
public:
	using Generator::Generator;

	template<class T>
	PTX::Address<B, T, PTX::GlobalSpace>* Generate(const PTX::ParameterVariable<PTX::PointerType<B, T>> *variable)
	{
		auto tidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_tid->GetVariable("%tid"), PTX::VectorElement::X);
		auto temp_tidx = this->m_builder->AllocateRegister<PTX::UInt32Type, ResourceKind::Internal>("tidx");

		auto temp0 = this->m_builder->AllocateRegister<PTX::UIntType<B>, ResourceKind::Internal>("0");
		auto temp1 = this->m_builder->AllocateRegister<PTX::UIntType<B>, ResourceKind::Internal>("1");
		auto temp2 = this->m_builder->AllocateRegister<PTX::UIntType<B>, ResourceKind::Internal>("2");
		auto temp3 = this->m_builder->AllocateRegister<PTX::UIntType<B>, ResourceKind::Internal>("3");

		auto temp0_ptr = new PTX::PointerRegisterAdapter<B, T>(temp0);
		auto temp1_ptr = new PTX::PointerRegisterAdapter<B, T, PTX::GlobalSpace>(temp1);
		auto temp3_ptr = new PTX::PointerRegisterAdapter<B, T, PTX::GlobalSpace>(temp3);

		this->m_builder->AddStatement(new PTX::LoadInstruction<B, PTX::PointerType<B, T>, PTX::ParameterSpace>(temp0_ptr, new PTX::MemoryAddress<B, PTX::PointerType<B, T>, PTX::ParameterSpace>(variable)));
		this->m_builder->AddStatement(new PTX::ConvertToAddressInstruction<B, T, PTX::GlobalSpace>(temp1_ptr, new PTX::RegisterAddress<B, T>(temp0_ptr)));
		this->m_builder->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(temp_tidx, tidx));
		if constexpr(B == PTX::Bits::Bits32)
		{
			auto temp2_bc = new PTX::Bit32RegisterAdapter<PTX::UIntType>(temp2);
			auto tidx_bc = new PTX::Bit32RegisterAdapter<PTX::UIntType>(temp_tidx);
			this->m_builder->AddStatement(new PTX::ShiftLeftInstruction<PTX::Bit32Type>(temp2_bc, tidx_bc, new PTX::UInt32Value(std::log2(PTX::BitSize<T::TypeBits>::Size / 8))));
		}
		else
		{
			this->m_builder->AddStatement(new PTX::MultiplyWideInstruction<PTX::UInt32Type>(temp2, temp_tidx, new PTX::UInt32Value(PTX::BitSize<T::TypeBits>::Size / 8)));
		}
		this->m_builder->AddStatement(new PTX::AddInstruction<PTX::UIntType<B>>(temp3, temp1, temp2));
		return new PTX::RegisterAddress<B, T, PTX::GlobalSpace>(temp3_ptr);
	}
};

}
