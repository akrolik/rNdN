#pragma once

#include "PTX/Type.h"
#include "PTX/Operands/Variables/Register.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Generator.h"
#include "Codegen/Generators/TargetGenerator.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"

namespace Codegen {

template<PTX::Bits B, class T>
class BuiltinGenerator : public Generator
{
public:
	using Generator::Generator;

	virtual void Generate(const HorseIR::LValue *target, const HorseIR::CallExpression *call)
	{
		// The default behaviour for a builtin generator creates a target register
		// of the correct type and generates the corresponding operation. This may
		// be overridden by replacing this call
		//
		// Depending on the input arguments, the output value may be compressed. If so,
		// the generator must provide a single compression predicate

		TargetGenerator<B, T> targetGenerator(this->m_builder);
		auto targetRegister = targetGenerator.Generate(target, GenerateCompressionPredicate(call));
		Generate(targetRegister, call);
	}

	virtual const PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const HorseIR::CallExpression *call)
	{
		// Default has no compression predicate

		return nullptr;
	}

	virtual void Generate(const PTX::Register<T> *target, const HorseIR::CallExpression *call)
	{
		Unimplemented(call);
	}

	[[noreturn]] static void Unimplemented(const HorseIR::CallExpression *call)
	{
		Unimplemented("builtin function " + HorseIR::PrettyPrinter::PrettyString(call->GetFunctionLiteral()));
	}

	[[noreturn]] static void Unimplemented(const std::string& context)
	{
		Utils::Logger::LogError("Generator does not support type " + T::Name() + " for " + context);
	}

};

}
