#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"
#include "Codegen/Generators/TypeDispatch.h"

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "PTX/Tree/Tree.h"

namespace Codegen {

template<PTX::Bits B>
class DeclarationGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "DeclarationGenerator"; }

	void Generate(const HorseIR::VariableDeclaration *declaration)
	{
		DispatchType(*this, declaration->GetType(), declaration);
	}

	template<class T>
	void GenerateVector(const HorseIR::VariableDeclaration *declaration)
	{
		// Allocate a new variable for each declaration

		auto resources = this->m_builder.GetLocalResources();
		auto name = NameUtils::VariableName(declaration);
		resources->template AllocateRegister<T>(name);
	}

	template<class T>
	void GenerateList(const HorseIR::VariableDeclaration *declaration)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Check for the type of load geometry

		auto& inputOptions = this->m_builder.GetInputOptions();
		if (inputOptions.IsVectorGeometry())
		{
			// List-in-vector declarations are handled separately

			const auto& shape = inputOptions.DeclarationShapes.at(declaration);
			if (const auto listShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(shape))
			{
				if (const auto size = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(listShape->GetListSize()))
				{
					for (auto index = 0u; index < size->GetValue(); ++index)
					{
						auto name = NameUtils::VariableName(declaration, index);
						resources->template AllocateRegister<T>(name);
					}
				}
				else
				{
					Error(declaration);
				}
			}
			else
			{
				Error(declaration);
			}
		}
		else if (inputOptions.IsListGeometry())
		{
			auto name = NameUtils::VariableName(declaration);
			resources->template AllocateRegister<T>(name);
		}
		else
		{
			Error(declaration);
		}
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::VariableDeclaration *declaration)
	{
		if (!this->m_builder.GetInputOptions().IsVectorGeometry())
		{
			Error(declaration);
		}

		auto resources = this->m_builder.GetLocalResources();
		auto name = NameUtils::VariableName(declaration, index);
		resources->template AllocateRegister<T>(name);
	}

	[[noreturn]] void Error(const HorseIR::VariableDeclaration *declaration)
	{
		Generator::Error("variable declaration " + HorseIR::PrettyPrinter::PrettyString(declaration, true));
	}
};

}
