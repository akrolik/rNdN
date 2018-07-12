#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/ResourceAllocator.h"
#include "Codegen/Generators/Expressions/ExpressionGenerator.h"

#include "HorseIR/Tree/Statements/AssignStatement.h"

#include "PTX/Type.h"
#include "PTX/Functions/Function.h"
#include "PTX/Operands/Variables/Register.h"

namespace Codegen {

template<PTX::Bits B>
class AssignmentGenerator : public Generator
{
public:
	using Generator::Generator;

	template<class T>
	void Generate(HorseIR::AssignStatement *assign)
	{
		// An assignment in HorseIR consists of: name, type, and expression (typically a function call).
		// This presents a small difficulty since PTX is 3-address code and links together all 3 elements
		// into a single instruction. We therefore can't completely separate the expression generation
		// from the assignment as is typically done in McLab compilers.
		//
		// (1) First generate the target register for the assignment destination type T
		// (2) Create a typed expression visitor (using the destination type) to evaluate the RHS of
		//     the assignment. We assume that the RHS and target have the same types
		// (3) Visit the expression
		//
		// In this setup, the expression visitor is expected to produce the full assignment

		const PTX::Register<T> *target = this->m_builder->template AllocateRegister<T>(assign->GetTargetName());
		ExpressionGenerator<B, T> generator(target, this->m_builder);
		assign->Accept(generator);
	}
};

}
