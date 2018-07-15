#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "PTX/Statements/CommentStatement.h"

namespace Codegen {

enum class ReductionOperation {
	Count,
	Sum,
	Average,
	Minimum,
	Maximum
};

template<PTX::Bits B, class T>
class ReductionGenerator : public BuiltinGenerator<B, T>
{
public:
	ReductionGenerator(Builder *builder, ReductionOperation reductionOp) : BuiltinGenerator<B, T>(builder), m_reductionOp(reductionOp) {}

	void Generate(const PTX::Register<T> *target, const HorseIR::CallExpression *call) override
	{
		//TODO: Implement @reduction builtin functions
		this->m_builder->AddStatement(new PTX::CommentStatement("<" + call->GetName() + ">"));
		// switch (m_reductionOp)
		// {
		// 	default:
		// 		this->Unimplemented(call);
		// }
	}

private:
	ReductionOperation m_reductionOp;
};

}
