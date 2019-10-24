#pragma once

#include <vector>

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/TargetGenerator.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "PTX/PTX.h"

#include "Utils/Logger.h"

namespace Codegen {

template<PTX::Bits B, class T>
class BuiltinGenerator : public Generator
{
public:
	using Generator::Generator;

	virtual const PTX::Register<T> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		Unimplemented();
	}

	const PTX::Register<T> *GenerateTargetRegister(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		// Depending on the input arguments, the output value may be compressed. If so,
		// the generator must provide a single compression predicate

		TargetGenerator<B, T> targetGenerator(this->m_builder);
		return targetGenerator.Generate(target, GenerateCompressionPredicate(arguments));
	}

	virtual const PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<HorseIR::Operand *>& arguments)
	{
		// Default has no compression predicate

		return nullptr;
	}

	[[noreturn]] static void Unimplemented()
	{
		Utils::Logger::LogError("Generator does not support type " + T::Name());
	}

	[[noreturn]] static void Unimplemented(const std::string& context)
	{
		Utils::Logger::LogError("Generator does not support type " + T::Name() + " for " + context);
	}

};

}
