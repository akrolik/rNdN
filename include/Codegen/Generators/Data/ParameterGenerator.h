#pragma once

#include "Codegen/Generators/Generator.h"

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
			auto shape = inputOptions.ParameterShapes.at(parameter->GetSymbol());
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

				GeneratePointer<T>(name);

				// Determine if we need a size parameter for the argument

				if (Runtime::RuntimeUtils::IsDynamicDataShape(shape, inputOptions.ThreadGeometry))
				{
					if (returnParameter)
					{
						// Return dynamic shapes use pointer types for accumulating size

						GeneratePointer<PTX::UInt32Type>(NameUtils::SizeName(name));
					}
					else
					{
						GenerateConstant<PTX::UInt32Type>(NameUtils::SizeName(name));
					}
				}
			}
			else if (Analysis::ShapeUtils::IsShape<Analysis::ListShape>(shape))
			{
				// Add list parameter (pointer of pointers) to the kernel

				GeneratePointer<PTX::PointerType<B, T, PTX::GlobalSpace>>(name);

				// Determine if we need a size parameter for the argument. List sizes are always stored in gobal space

				if (Runtime::RuntimeUtils::IsDynamicDataShape(shape, inputOptions.ThreadGeometry))
				{
					GeneratePointer<PTX::UInt32Type>(NameUtils::SizeName(name));
				}
			}
		}
	}

	template<class T>
	void GeneratePointer(const std::string& name)
	{
		// Allocate a pointer parameter declaration for the type

		this->m_builder.AddParameter(name, new PTX::PointerDeclaration<B, T>(name));
	}

	template<class T>
	void GenerateConstant(const std::string& name)
	{
		// Allocate a constant parameter declaration for the type

		this->m_builder.AddParameter(name, new PTX::ParameterDeclaration<T>(name));
	}
};

}