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
	ReductionGenerator(const PTX::Register<T> *target, Builder *builder, ReductionOperation reductionOp) : BuiltinGenerator<B, T>(target, builder), m_reductionOp(reductionOp) {}

	void Generate(const HorseIR::CallExpression *call) override
	{
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
