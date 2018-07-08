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

	virtual void Generate(const HorseIR::CallExpression *call) = 0;

protected:
	const PTX::Register<T> *m_target = nullptr;
	Builder *m_builder = nullptr;
};

}
