#pragma once

#include "Frontend/Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Expressions/ConversionGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataIndexGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B, class T>
class RangeGenerator : public BuiltinGenerator<B, T>
{
public:
	using BuiltinGenerator<B, T>::BuiltinGenerator;

	std::string Name() const override { return "RangeGenerator"; }
};

template<PTX::Bits B>
class RangeGenerator<B, PTX::Int64Type> : public BuiltinGenerator<B, PTX::Int64Type>
{
public:
	using BuiltinGenerator<B, PTX::Int64Type>::BuiltinGenerator;

	std::string Name() const override { return "RangeGenerator"; }

	// The output of the range function has no compression predicate. We therefore do not implement GenerateCompressionPredicate in this subclass

	PTX::Register<PTX::Int64Type> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		// Get the data index depending on the geometry

		DataIndexGenerator<B> indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateDataIndex();

		auto targetRegister = this->GenerateTargetRegister(target, arguments);
		ConversionGenerator::ConvertSource<PTX::Int64Type, PTX::UInt32Type>(this->m_builder, targetRegister, index);
		return targetRegister;
	}
};

}
}
