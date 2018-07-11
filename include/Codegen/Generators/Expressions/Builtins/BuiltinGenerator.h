#pragma once

#include "PTX/Type.h"
#include "PTX/Operands/Variables/Register.h"

#include "Codegen/Builder.h"

namespace Codegen {

template<PTX::Bits B, class T>
class BuiltinGenerator
{
public:
	BuiltinGenerator(const PTX::Register<T> *target, Builder *builder) : m_target(target), m_builder(builder) {}

	virtual void Generate(const HorseIR::CallExpression *call)
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

protected:
	const PTX::Register<T> *m_target = nullptr;
	Builder *m_builder = nullptr;
};

}
