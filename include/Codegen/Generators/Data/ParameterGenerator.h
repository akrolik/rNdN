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

	std::string Name() const override { return "ParameterGenerator"; }

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
	void GenerateVector(const std::string& name, const Analysis::Shape *shape, bool returnParameter = false)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			// Predicate values are stored as 8-bit integers on the CPU

			GenerateVector<PTX::Int8Type>(name, shape, returnParameter);
		}
		else
		{
			// Add vector parameter to the kernel

			auto parameter = GeneratePointer<T>(name);
			if (returnParameter)
			{
				// Return dynamic shapes use a pointer parameter for accumulating size

				auto& inputOptions = this->m_builder.GetInputOptions();
				if (Runtime::RuntimeUtils::IsDynamicReturnShape(shape, inputOptions.ThreadGeometry))
				{
					GeneratePointer<PTX::UInt32Type>(NameUtils::SizeName(parameter));
				}
			}
			else
			{
				// Input parameters always have a size argument

				GenerateConstant<PTX::UInt32Type>(NameUtils::SizeName(parameter));
			}
		}
	}

	template<class T>
	void GenerateList(const std::string& name, const Analysis::Shape *shape, bool returnParameter = false)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			// Predicate values are stored as 8-bit integers on the CPU

			GenerateList<PTX::Int8Type>(name, shape, returnParameter);
		}
		else
		{
			// Add list parameter (pointer of pointers) to the kernel

			auto parameter = GeneratePointer<PTX::PointerType<B, T, PTX::GlobalSpace>>(name);

			// Dynamic returns as well as all input parameters require a size argument

			auto& inputOptions = this->m_builder.GetInputOptions();
			if (!returnParameter || Runtime::RuntimeUtils::IsDynamicReturnShape(shape, inputOptions.ThreadGeometry))
			{
				GeneratePointer<PTX::UInt32Type>(NameUtils::SizeName(parameter));
			}
		}
	}

	template<class T>
	void GenerateTuple(unsigned int index, const std::string& name, const Analysis::Shape *shape, bool returnParameter = false)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			// Predicate values are stored as 8-bit integers on the CPU

			GenerateTuple<PTX::Int8Type>(index, name, shape, returnParameter);
		}
		else
		{
			if (!this->m_builder.GetInputOptions().IsVectorGeometry())
			{
				Error(name, shape, returnParameter);
			}

			// Add list parameter to the kernel, aliasing all later types

			auto parameter = GeneratePointer<PTX::PointerType<B, T, PTX::GlobalSpace>>(name, index > 0);

			if (index == 0)
			{
				// Dynamic returns as well as all input parameters require a size argument

				auto& inputOptions = this->m_builder.GetInputOptions();
				if (!returnParameter || Runtime::RuntimeUtils::IsDynamicReturnShape(shape, inputOptions.ThreadGeometry))
				{
					GeneratePointer<PTX::UInt32Type>(NameUtils::SizeName(parameter));
				}
			}
		}
	}

	template<class T>
	const PTX::ParameterVariable<PTX::PointerType<B, T>> *GeneratePointer(const std::string& name, bool alias = false)
	{
		// Allocate a pointer parameter declaration for the type

		auto declaration = new PTX::PointerDeclaration<B, T>(name);
		this->m_builder.AddParameter(name, declaration, alias);

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

	[[noreturn]] void Error(const std::string& name, const Analysis::Shape *shape, bool returnParameter = false) const
	{
		if (returnParameter)
		{
			Generator::Error("return parameter '" + name + "' for shape " + Analysis::ShapeUtils::ShapeString(shape));
		}
		Generator::Error("parameter '" + name + "' for shape " + Analysis::ShapeUtils::ShapeString(shape));
	}
};

}
