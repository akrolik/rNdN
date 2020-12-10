#pragma once

#include "Frontend/Codegen/Generators/Generator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/NameUtils.h"

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

#include "Runtime/RuntimeUtils.h"

namespace Frontend {
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
			DispatchType(*this, parameter->GetType(), parameter->GetName(), shape);
		}

		auto returnIndex = 0;
		for (const auto& returnType : function->GetReturnTypes())
		{
			auto shape = inputOptions.ReturnShapes.at(returnIndex);
			auto writeShape = inputOptions.ReturnWriteShapes.at(returnIndex);

			auto name = NameUtils::ReturnName(returnIndex);
			DispatchType(*this, returnType, name, shape, writeShape);

			returnIndex++;
		}
	}

	template<class T>
	void GenerateVector(const std::string& name, const HorseIR::Analysis::Shape *shape, const HorseIR::Analysis::Shape *writeShape = nullptr)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			// Predicate values are stored as 8-bit integers on the CPU

			GenerateVector<PTX::Int8Type>(name, shape, writeShape);
		}
		else
		{
			// Add vector parameter to the kernel

			auto parameter = GeneratePointer<T>(name);
			GeneratePointer<PTX::UInt32Type>(NameUtils::SizeName(parameter));
		}
	}

	template<class T>
	void GenerateList(const std::string& name, const HorseIR::Analysis::Shape *shape, const HorseIR::Analysis::Shape *writeShape = nullptr)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			// Predicate values are stored as 8-bit integers on the CPU

			GenerateList<PTX::Int8Type>(name, shape, writeShape);
		}
		else
		{
			// Add list parameter (pointer of pointers) to the kernel

			auto parameter = GeneratePointer<PTX::PointerType<B, T, PTX::GlobalSpace>>(name);
			GeneratePointer<PTX::PointerType<B, PTX::UInt32Type, PTX::GlobalSpace>>(NameUtils::SizeName(parameter));
		}
	}

	template<class T>
	void GenerateTuple(unsigned int index, const std::string& name, const HorseIR::Analysis::Shape *shape, const HorseIR::Analysis::Shape *writeShape = nullptr)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			// Predicate values are stored as 8-bit integers on the CPU

			GenerateTuple<PTX::Int8Type>(index, name, shape, writeShape);
		}
		else
		{
			if (!this->m_builder.GetInputOptions().IsVectorGeometry())
			{
				Error(name, shape, (writeShape != nullptr));
			}

			// Add list parameter to the kernel, aliasing all later types

			auto parameter = GeneratePointer<PTX::PointerType<B, T, PTX::GlobalSpace>>(name, index > 0);

			if (index == 0)
			{
				// Double indirection size parameters

				GeneratePointer<PTX::PointerType<B, PTX::UInt32Type, PTX::GlobalSpace>>(NameUtils::SizeName(parameter));
			}
		}
	}

	template<class T>
	PTX::ParameterVariable<T> *GenerateConstant(const std::string& name, bool alias = false)
	{
		// Allocate a constant parameter declaration for the type

		auto declaration = new PTX::ParameterDeclaration<T>(name);
		this->m_builder.AddParameter(name, declaration, alias);

		return declaration->GetVariable(name);
	}

	template<class T>
	PTX::ParameterVariable<PTX::PointerType<B, T>> *GeneratePointer(const std::string& name, bool alias = false)
	{
		// Allocate a pointer parameter declaration for the type

		auto declaration = new PTX::PointerDeclaration<B, T>(name);
		this->m_builder.AddParameter(name, declaration, alias);

		return declaration->GetVariable(name);
	}

	[[noreturn]] void Error(const std::string& name, const HorseIR::Analysis::Shape *shape, bool returnParameter = false) const
	{
		if (returnParameter)
		{
			Generator::Error("return parameter '" + name + "' for shape " + HorseIR::Analysis::ShapeUtils::ShapeString(shape));
		}
		Generator::Error("parameter '" + name + "' for shape " + HorseIR::Analysis::ShapeUtils::ShapeString(shape));
	}
};

}
}
