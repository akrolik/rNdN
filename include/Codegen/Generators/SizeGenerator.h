#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"

#include "PTX/Type.h"
#include "PTX/Declarations/SpecialRegisterDeclarations.h"
#include "PTX/Instructions/Arithmetic/DivideInstruction.h"
#include "PTX/Instructions/Data/MoveInstruction.h"
#include "PTX/Operands/SpecialConstants.h"
#include "PTX/Operands/Variables/IndexedRegister.h"

namespace Codegen {

class SizeGenerator : public Generator
{
public:
	using Generator::Generator;

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateWarpCount()
	{
		auto srntidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_ntid->GetVariable("%ntid"), PTX::VectorElement::X);

		// We cannot operate directly on special registers, so they must first be copied to a user defined register

		auto ntidx = this->m_builder->template AllocateTemporary<PTX::UInt32Type>();
		auto count = this->m_builder->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(ntidx, srntidx));
		this->m_builder->AddStatement(new PTX::DivideInstruction<PTX::UInt32Type>(count, ntidx, PTX::SpecialConstant_WARP_SZ));

		return count;
	}
};

}
