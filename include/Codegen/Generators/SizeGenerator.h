#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"

#include "PTX/PTX.h"

namespace Codegen {

class SizeGenerator : public Generator
{
public:
	using Generator::Generator;

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateVectorSize(const std::string& name)
	{
		//TODO: Statically determined sizes
		// Get the special size register for the vector (assumed dynamic input)

		auto resources = this->m_builder.GetLocalResources();
		return resources->GetRegister<PTX::UInt32Type>(NameUtils::SizeName(name));
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateCellSize(const std::string& name)
	{
		//TODO: Statically determined sizes
		// Get the special size register for the list (assumes dynamic input)

		auto resources = this->m_builder.GetLocalResources();
		return resources->GetRegister<PTX::UInt32Type>(NameUtils::SizeName(name));
	}
};

}
