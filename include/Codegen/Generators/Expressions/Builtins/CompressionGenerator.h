#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "PTX/Statements/CommentStatement.h"

namespace Codegen {

template<PTX::Bits B, class T>
class CompressionGenerator : public BuiltinGenerator<B, T>
{
public:
	using BuiltinGenerator<B, T>::BuiltinGenerator;

	void Generate(const PTX::Register<T> *target, const HorseIR::CallExpression *call) override
	{
		//TODO: Implement @compress builtin
		this->m_builder->AddStatement(new PTX::CommentStatement("<compress>"));
	}
};

}
