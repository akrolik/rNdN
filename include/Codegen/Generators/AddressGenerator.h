#pragma once

#include <cmath>

#include "Codegen/GeneratorState.h"
#include "Codegen/Generators/AddressGenerator.h"

#include "HorseIR/Tree/Statements/ReturnStatement.h"

#include "PTX/Type.h"
#include "PTX/Declarations/SpecialRegisterDeclarations.h"
#include "PTX/Functions/Function.h"
#include "PTX/Instructions/Arithmetic/AddInstruction.h"
#include "PTX/Instructions/Arithmetic/MultiplyWideInstruction.h"
#include "PTX/Instructions/Data/ConvertToAddressInstruction.h"
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
#include "HorseIR/Tree/Statements/ReturnStatement.h"

#include "PTX/StateSpace.h"
#include "PTX/Type.h"
#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Variables/AddressableVariable.h"

template<PTX::Bits B>
class AddressGenerator
{
public:
	template<class T>
	static PTX::Address<B, T, PTX::GlobalSpace>* Generate(const PTX::ParameterVariable<PTX::PointerType<T, B>> *variable, GeneratorState *state)
	{
		auto resources = state->GetCurrentResources();
		auto block = state->GetCurrentBlock();

		auto tidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_tid->GetVariable("%tid"), PTX::VectorElement::X);
		auto temp_tidx = resources->template AllocateRegister<PTX::UInt32Type, ResourceType::Temporary>("tidx");

		auto temp0 = resources->template AllocateRegister<PTX::UIntType<B>, ResourceType::Temporary>("0");
		auto temp1 = resources->template AllocateRegister<PTX::UIntType<B>, ResourceType::Temporary>("1");
		auto temp2 = resources->template AllocateRegister<PTX::UIntType<B>, ResourceType::Temporary>("2");
		auto temp3 = resources->template AllocateRegister<PTX::UIntType<B>, ResourceType::Temporary>("3");

		auto temp0_ptr = new PTX::PointerRegisterAdapter<T, B>(temp0);
		auto temp1_ptr = new PTX::PointerRegisterAdapter<T, B, PTX::GlobalSpace>(temp1);
		auto temp3_ptr = new PTX::PointerRegisterAdapter<T, B, PTX::GlobalSpace>(temp3);

		block->AddStatement(new PTX::Load64Instruction<PTX::PointerType<T, B>, PTX::ParameterSpace>(temp0_ptr, new PTX::MemoryAddress64<PTX::PointerType<T, B>, PTX::ParameterSpace>(variable)));
		block->AddStatement(new PTX::ConvertToAddressInstruction<T, B, PTX::GlobalSpace>(temp1_ptr, temp0_ptr));
		block->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(temp_tidx, tidx));
		if constexpr(B == PTX::Bits::Bits32)
		{
			auto temp2_bc = new PTX::Bit32RegisterAdapter<PTX::UIntType>(temp2);
			auto tidx_bc = new PTX::Bit32RegisterAdapter<PTX::UIntType>(temp_tidx);
			block->AddStatement(new PTX::ShiftLeftInstruction<PTX::Bit32Type>(temp2_bc, tidx_bc, new PTX::UInt32Value(std::log2(T::BitSize / 8))));
		}
		else
		{
			block->AddStatement(new PTX::MultiplyWideInstruction<PTX::UIntType<B>, PTX::UInt32Type>(temp2, temp_tidx, new PTX::UInt32Value(T::BitSize / 8)));
		}
		block->AddStatement(new PTX::AddInstruction<PTX::UIntType<B>>(temp3, temp1, temp2));
		return new PTX::RegisterAddress<B, T, PTX::GlobalSpace>(temp3_ptr);
	}
};
