#pragma once

#include "Codegen/GeneratorState.h"
#include "Codegen/ResourceAllocator.h"
#include "Codegen/Generators/Expressions/ExpressionGenerator.h"

#include "HorseIR/Tree/Statements/AssignStatement.h"

#include "PTX/Type.h"
#include "PTX/Functions/Function.h"
#include "PTX/Operands/Variables/Register.h"

template<PTX::Bits B>
class AssignmentGenerator
{
public:
	using NodeType = HorseIR::AssignStatement;

	template<class T>
	static void Generate(HorseIR::AssignStatement *assign, GeneratorState *state)
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

		auto function = state->GetCurrentFunction();
		auto resources = state->GetCurrentResources();

		const PTX::Register<T> *target = resources->template AllocateRegister<T>(assign->GetIdentifier());
		ExpressionGenerator<B, T> generator(target, function, resources);
		assign->Accept(generator);
	}
};
