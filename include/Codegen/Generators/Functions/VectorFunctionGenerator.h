#pragma once

#include "Codegen/Generators/Functions/FunctionGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"
#include "Codegen/Generators/Data/ParameterGenerator.h"
#include "Codegen/Generators/Data/ParameterLoadGenerator.h"
#include "Codegen/Generators/Data/ValueLoadGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class VectorFunctionGenerator : public FunctionGenerator<B>
{
public:
	using FunctionGenerator<B>::FunctionGenerator;

	void Generate(const HorseIR::Function *function)
	{
		// Initialize the parameters from the address space

		ParameterLoadGenerator<B> parameterLoadGenerator(this->m_builder);
		for (const auto& parameter : function->GetParameters())
		{
			parameterLoadGenerator.Generate(parameter);
		}
		parameterLoadGenerator.Generate(function->GetReturnTypes());

		auto& inputOptions = this->m_builder.GetInputOptions();

		// Check if the geometry size is dynamically specified
		
		const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(inputOptions.ThreadGeometry);
		if (Analysis::ShapeUtils::IsDynamicSize(vectorGeometry->GetSize()))
		{
			// Load the geometry size from the input

			this->m_builder.AddStatement(new PTX::CommentStatement(NameUtils::GeometryVectorSize));

			ParameterGenerator<B> parameterGenerator(this->m_builder);
			parameterGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::GeometryVectorSize);

			ValueLoadGenerator<B> valueLoadGenerator(this->m_builder);
			valueLoadGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::GeometryVectorSize);
		}

		for (const auto& statement : function->GetStatements())
		{
			statement->Accept(*this);
		}
	}

	void Visit(const HorseIR::ReturnStatement *returnS) override
	{
		// Finalize the vector function return statement by a return instruction

		FunctionGenerator<B>::Visit(returnS);
		this->m_builder.AddStatement(new PTX::ReturnInstruction());
	}
};

}
