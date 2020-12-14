#pragma once

#include <vector>

#include "Frontend/Codegen/Generators/Generator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Data/TargetGenerator.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B, class T>
class BuiltinGenerator : public Generator
{
public:
	using Generator::Generator;

	virtual PTX::Register<T> *Generate(const HorseIR::LValue *target, const std::vector<const HorseIR::Operand *>& arguments)
	{
		Unimplemented();
	}

	PTX::Register<T> *GenerateTargetRegister(const HorseIR::LValue *target, const std::vector<const HorseIR::Operand *>& arguments)
	{
		// Depending on the input arguments, the output value may be compressed. If so,
		// the generator must provide a single compression predicate

		TargetGenerator<B, T> targetGenerator(this->m_builder);
		return targetGenerator.Generate(target, GenerateCompressionPredicate(arguments));
	}

	virtual PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<const HorseIR::Operand *>& arguments)
	{
		// Default has no compression predicate

		return nullptr;
	}

	[[noreturn]] void Unimplemented() const
	{
		Error("type " + T::Name());
	}

	[[noreturn]] void Unimplemented(const std::string& context) const
	{
		Error("type " + T::Name() + " for " + context);
	}

};

}
}
