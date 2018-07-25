#pragma once

#include <cmath>

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"

#include "PTX/Type.h"
#include "PTX/Declarations/SpecialRegisterDeclarations.h"
#include "PTX/Instructions/Arithmetic/MADInstruction.h"
#include "PTX/Instructions/Data/MoveInstruction.h"
#include "PTX/Operands/Variables/IndexedRegister.h"

namespace Codegen {

class IndexGenerator : public Generator
{
public:
	using Generator::Generator;

	const PTX::Register<PTX::UInt32Type> *GenerateLocalIndex()
	{
		auto srtidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_tid->GetVariable("%tid"), PTX::VectorElement::X);

		// We cannot operate directly on special registers, so they must first be copied to a user defined register

		auto tidx = this->m_builder->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(tidx, srtidx));

		return tidx;
	}

	const PTX::Register<PTX::UInt32Type> *GenerateGlobalIndex()
	{
		auto srtidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_tid->GetVariable("%tid"), PTX::VectorElement::X);
		auto srctaidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_ctaid->GetVariable("%ctaid"), PTX::VectorElement::X);
		auto srntidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_ntid->GetVariable("%ntid"), PTX::VectorElement::X);

		// We cannot operate directly on special registers, so they must first be copied to a user defined register

		auto tidx = this->m_builder->template AllocateTemporary<PTX::UInt32Type>();
		auto ctaidx = this->m_builder->template AllocateTemporary<PTX::UInt32Type>();
		auto ntidx = this->m_builder->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(tidx, srtidx));
		this->m_builder->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(ctaidx, srctaidx));
		this->m_builder->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(ntidx, srntidx));

		// Compute the thread index as blockSize * blockIndex + threadIndex

		auto index = this->m_builder->template AllocateTemporary<PTX::UInt32Type>("index");

		auto madInstruction = new PTX::MADInstruction<PTX::UInt32Type>(index, ctaidx, ntidx, tidx);
		madInstruction->SetLower(true);
		this->m_builder->AddStatement(madInstruction);

		return index;
	}
};

}
