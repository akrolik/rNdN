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
