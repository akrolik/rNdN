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
};

template<PTX::Bits B>
class IndexOfGenerator<B, PTX::Int64Type>: public BuiltinGenerator<B, PTX::Int64Type>
{
public:
	using BuiltinGenerator<B, PTX::Int64Type>::BuiltinGenerator;

	const PTX::Register<PTX::Int64Type> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		DispatchType(*this, arguments.at(0)->GetType(), target, arguments);
		return m_targetRegister;
	}

	template<typename T>
	void Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		//TODO: Use comparison generator instead
		if constexpr(std::is_same<T, PTX::Int8Type>::value)
		{
			// Comparison can only occur for 16-bits+

			Generate<PTX::Int16Type>(target, arguments);
		}
		else
		{
			InternalFindGenerator<B, T, PTX::Int64Type> findGenerator(this->m_builder);
			m_targetRegister = findGenerator.Generate(target, arguments);
		}
	}

private:
	const PTX::Register<PTX::Int64Type> *m_targetRegister = nullptr;
};

}
