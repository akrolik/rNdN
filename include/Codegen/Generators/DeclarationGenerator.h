#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/Resources/RegisterAllocator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class DeclarationGenerator : public Generator
{
public:
	using Generator::Generator;

	template<class T>
	void Generate(const HorseIR::VariableDeclaration *declaration)
	{
		auto resources = this->m_builder.GetLocalResources();
		resources->template AllocateRegister<T>(declaration->GetName());
	}
};

}
