#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/OperandCompressionGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/InternalFindGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B, class T>
class IndexOfGenerator : public BuiltinGenerator<B, T>
{
public: 
	using BuiltinGenerator<B, T>::BuiltinGenerator;

	std::string Name() const override { return "IndexOfGenerator"; }
};

template<PTX::Bits B>
class IndexOfGenerator<B, PTX::Int64Type>: public BuiltinGenerator<B, PTX::Int64Type>
{
public:
	using BuiltinGenerator<B, PTX::Int64Type>::BuiltinGenerator;

	std::string Name() const override { return "IndexOfGenerator"; }

	const PTX::Register<PTX::Int64Type> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		DispatchType(*this, arguments.at(0)->GetType(), target, arguments);
		return m_targetRegister;
	}

	template<typename T>
	void GenerateVector(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		//TODO: Use comparison generator instead
		if constexpr(std::is_same<T, PTX::Int8Type>::value)
		{
			// Comparison can only occur for 16-bits+

			GenerateVector<PTX::Int16Type>(target, arguments);
		}
		else
		{
			InternalFindGenerator<B, T, PTX::Int64Type> findGenerator(this->m_builder);
			m_targetRegister = findGenerator.Generate(target, arguments);
		}
	}

	template<typename T>
	void GenerateList(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		if (this->m_builder.GetInputOptions().IsVectorGeometry())
		{
			BuiltinGenerator<B, PTX::Int64Type>::Unimplemented("list-in-vector");
		}

		// Lists are handled by the vector code through a projection

		GenerateVector<T>(target, arguments);
	}

	template<typename T>
	void GenerateTuple(unsigned int index, const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		BuiltinGenerator<B, PTX::Int64Type>::Unimplemented("list-in-vector");
	}

private:
	const PTX::Register<PTX::Int64Type> *m_targetRegister = nullptr;
};

}
