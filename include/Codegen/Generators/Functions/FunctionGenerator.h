#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"
#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/DeclarationGenerator.h"
#include "Codegen/Generators/Data/ValueStoreGenerator.h"
#include "Codegen/Generators/Expressions/ExpressionGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class FunctionGenerator : public HorseIR::ConstVisitor, public Generator
{
public:
	using Generator::Generator;

	void Visit(const HorseIR::DeclarationStatement *declarationS) override
	{
		DeclarationGenerator<B> generator(m_builder);
		generator.Generate(declarationS->GetDeclaration());
	}

	void Visit(const HorseIR::AssignStatement *assignS) override
	{
		// An assignment in HorseIR consists of: one or more LValues and expression (typically a function call).
		// This presents a small difficulty since PTX is 3-address code and links together all 3 elements into
		// a single instruction. Additionally, for functions with more than one return value, it is challenging
		// to maintain static type-correctness.
		//
		// In this setup, the expression visitor is expected to produce the full assignment

		m_builder.AddStatement(new PTX::CommentStatement(HorseIR::PrettyPrinter::PrettyString(assignS, true)));

		ExpressionGenerator<B> generator(m_builder);
		generator.Generate(assignS->GetTargets(), assignS->GetExpression());
	}

	void Visit(const HorseIR::ExpressionStatement *expressionS) override
	{
		// Expression generator may also take zero targets and discard the resulting value

		m_builder.AddStatement(new PTX::CommentStatement(HorseIR::PrettyPrinter::PrettyString(expressionS, true)));

		ExpressionGenerator<B> generator(m_builder);
		generator.Generate(expressionS->GetExpression());
	}

	void Visit(const HorseIR::ReturnStatement *returnS) override
	{
		// Return generator generates the output results (atomic, compressed, or list/vector)

		m_builder.AddStatement(new PTX::CommentStatement(HorseIR::PrettyPrinter::PrettyString(returnS, true)));

		ValueStoreGenerator<B> generator(m_builder);
		generator.Generate(returnS);
	}
};

}