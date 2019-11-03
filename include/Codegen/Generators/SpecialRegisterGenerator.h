#pragma once

#include "Codegen/Generators/Generator.h"

#include "PTX/PTX.h"

namespace Codegen {

class SpecialRegisterGenerator : public Generator
{
public:
	using Generator::Generator;

	const PTX::Register<PTX::UInt32Type> *GenerateThreadIndex(PTX::VectorElement element = PTX::VectorElement::X)
	{
		return GenerateSpecialRegister(PTX::SpecialRegisterDeclaration_tid->GetVariable("%tid"), element);
	}

	const PTX::Register<PTX::UInt32Type> *GenerateThreadCount(PTX::VectorElement element = PTX::VectorElement::X)
	{
		return GenerateSpecialRegister(PTX::SpecialRegisterDeclaration_ntid->GetVariable("%ntid"), element);
	}

	const PTX::Register<PTX::UInt32Type> *GenerateBlockIndex(PTX::VectorElement element = PTX::VectorElement::X)
	{
		return GenerateSpecialRegister(PTX::SpecialRegisterDeclaration_ctaid->GetVariable("%ctaid"), element);
	}

	const PTX::Register<PTX::UInt32Type> *GenerateBlockCount(PTX::VectorElement element = PTX::VectorElement::X)
	{
		return GenerateSpecialRegister(PTX::SpecialRegisterDeclaration_nctaid->GetVariable("%nctaid"), element);
	}

	const PTX::Register<PTX::UInt32Type> *GenerateSpecialRegister(const PTX::Register<PTX::Vector4Type<PTX::UInt32Type>> *specialRegister, PTX::VectorElement element)
	{
		auto resources = this->m_builder.GetLocalResources();

		// We cannot operate directly on special registers, so they must first be copied to a user defined register

		auto indexedRegister = new PTX::IndexedRegister4<PTX::UInt32Type>(specialRegister, element);
		auto valueRegister = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(valueRegister, indexedRegister));

		return valueRegister;
	}
};

}
