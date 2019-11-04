#pragma once

#include "Codegen/Generators/Generator.h"

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

#include "Runtime/RuntimeUtils.h"

namespace Codegen {

template<PTX::Bits B>
class ParameterGenerator : public Generator
{
public:
	using Generator::Generator;

	void Generate(const HorseIR::Function *function)
	{
		auto& inputOptions = this->m_builder.GetInputOptions();

		for (const auto& parameter : function->GetParameters())
		{
			auto shape = inputOptions.ParameterShapes.at(parameter);
			DispatchType(*this, parameter->GetType(), parameter->GetName(), shape, false);
		}

		auto returnIndex = 0;
		for (const auto& returnType : function->GetReturnTypes())
		{
			auto shape = inputOptions.ReturnShapes.at(returnIndex);
			auto name = NameUtils::ReturnName(returnIndex);
			DispatchType(*this, returnType, name, shape, true);

			returnIndex++;
		}
	}

	template<class T>
	void Generate(const std::string& name, const Analysis::Shape *shape, bool returnParameter)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			// Predicate values are stored as 8-bit integers on the CPU

			Generate<PTX::Int8Type>(name, shape, returnParameter);
		}
		else
		{
			auto& inputOptions = this->m_builder.GetInputOptions();
			if (Analysis::ShapeUtils::IsShape<Analysis::VectorShape>(shape))
			{
				// Add vector parameter to the kernel

				auto parameter = GeneratePointer<T>(name);

				// Determine if we need a size parameter for the argument

				if (Runtime::RuntimeUtils::IsDynamicDataShape(shape, inputOptions.ThreadGeometry, returnParameter))
				{
					if (returnParameter)
					{
						// Return dynamic shapes use pointer types for accumulating size

						GeneratePointer<PTX::UInt32Type>(NameUtils::SizeName(parameter));
					}
					else
					{
						GenerateConstant<PTX::UInt32Type>(NameUtils::SizeName(parameter));
					}
				}
			}
			else if (Analysis::ShapeUtils::IsShape<Analysis::ListShape>(shape))
			{
				// Add list parameter (pointer of pointers) to the kernel

				auto parameter = GeneratePointer<PTX::PointerType<B, T, PTX::GlobalSpace>>(name);

				// Determine if we need a size parameter for the argument. List sizes are always stored in global space

				if (Runtime::RuntimeUtils::IsDynamicDataShape(shape, inputOptions.ThreadGeometry, returnParameter))
				{
					GeneratePointer<PTX::UInt32Type>(NameUtils::SizeName(parameter));
				}
			}
		}
	}

	template<class T>
	const PTX::ParameterVariable<PTX::PointerType<B, T>> *GeneratePointer(const std::string& name)
	{
		// Allocate a pointer parameter declaration for the type

		auto declaration = new PTX::PointerDeclaration<B, T>(name);
		this->m_builder.AddParameter(name, declaration);

		return declaration->GetVariable(name);
	}

	template<class T>
	const PTX::ParameterVariable<T> *GenerateConstant(const std::string& name)
	{
		// Allocate a constant parameter declaration for the type

		auto declaration = new PTX::ParameterDeclaration<T>(name);
		this->m_builder.AddParameter(name, declaration);

		return declaration->GetVariable(name);
	}
};

}
