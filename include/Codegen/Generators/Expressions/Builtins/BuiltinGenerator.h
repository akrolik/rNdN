#pragma once

#include "PTX/Type.h"
#include "PTX/Operands/Variables/Register.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Generator.h"

namespace Codegen {

template<PTX::Bits B, class T>
class BuiltinGenerator : public Generator
{
public:
	using Generator::Generator;

	virtual void Generate(const std::string& target, const HorseIR::CallExpression *call)
	{
		// The default behaviour for a builtin generator creates a target register
		// of the correct type and generates the corresponding operation. This may
		// be overridden by replacing this call
		//
		// Depending on the input arguments, the output value may be compressed. If so,
		// the generator must provide a single compression predicate

		const PTX::Register<T> *targetRegister = this->m_builder->template AllocateRegister<T>(target, GenerateCompressionPredicate(call));
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
		Unimplemented("builtin function " + call->GetName());
	}

	[[noreturn]] static void Unimplemented(const std::string& context)
	{
		std::cerr << "[ERROR] Generator does not support type " << T::Name() << " for " << context << std::endl;
		std::exit(EXIT_FAILURE);
	}

};

}
