#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"
#include "Codegen/Generators/TypeDispatch.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class DeclarationGenerator : public Generator
{
public:
	using Generator::Generator;

	void Generate(const HorseIR::VariableDeclaration *declaration)
	{
		DispatchType(*this, declaration->GetType(), declaration);
	}

	template<class T>
	void Generate(const HorseIR::VariableDeclaration *declaration)
	{
		// Allocate a new variable for each declaration

		auto resources = this->m_builder.GetLocalResources();
		auto name = NameUtils::VariableName(declaration->GetName());
		resources->template AllocateRegister<T>(name);
	}
};

}
