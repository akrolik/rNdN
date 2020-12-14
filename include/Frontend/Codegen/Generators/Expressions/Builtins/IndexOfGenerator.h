#pragma once

#include "Frontend/Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Expressions/Builtins/InternalFindGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
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

	PTX::Register<PTX::Int64Type> *Generate(const HorseIR::LValue *target, const std::vector<const HorseIR::Operand *>& arguments) override
	{
		InternalFindGenerator<B, PTX::Int64Type> findGenerator(this->m_builder, FindOperation::Index, {ComparisonOperation::Equal});
		return findGenerator.Generate(target, arguments);
	}
};

}
}
