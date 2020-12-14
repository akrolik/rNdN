#pragma once

#include "Frontend/Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Expressions/Builtins/InternalFindGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B, class T>
class MemberGenerator : public BuiltinGenerator<B, T>
{
public: 
	using BuiltinGenerator<B, T>::BuiltinGenerator;

	std::string Name() const override { return "MemberGenerator"; }
};

template<PTX::Bits B>
class MemberGenerator<B, PTX::PredicateType>: public BuiltinGenerator<B, PTX::PredicateType>
{
public:
	using BuiltinGenerator<B, PTX::PredicateType>::BuiltinGenerator;

	std::string Name() const override { return "MemberGenerator"; }

	PTX::Register<PTX::PredicateType> *Generate(const HorseIR::LValue *target, const std::vector<const HorseIR::Operand *>& arguments) override
	{
		InternalFindGenerator<B, PTX::PredicateType> findGenerator(this->m_builder, FindOperation::Member, {ComparisonOperation::Equal});
		return findGenerator.Generate(target, arguments);
	}
};

}
}
