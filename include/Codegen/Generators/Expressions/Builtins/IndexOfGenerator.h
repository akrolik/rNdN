#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
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
		InternalFindGenerator<B, PTX::Int64Type> findGenerator(this->m_builder, FindOperation::Index, {ComparisonOperation::Equal});
		return findGenerator.Generate(target, arguments);
	}
};

}
