#include "Backend/Codegen/CodeGenerator.h"

#include "Backend/Codegen/Generators/InstructionGenerator.h"

namespace Backend {
namespace Codegen {

// Public API

SASS::Function *CodeGenerator::Generate(const PTX::FunctionDefinition<PTX::VoidType> *function, const PTX::Analysis::RegisterAllocation *allocation)
{
	auto sassFunction = m_builder.CreateFunction(function->GetName(), allocation);
	function->Accept(*this);
	m_builder.CloseFunction();
	return sassFunction;
}

// Declarations

bool CodeGenerator::VisitIn(const PTX::VariableDeclaration *declaration)
{
	declaration->Accept(static_cast<ConstDeclarationVisitor&>(*this));
	return false;
}

void CodeGenerator::Visit(const PTX::_TypedVariableDeclaration *declaration)
{
	declaration->Dispatch(*this);
}

template<class T, class S>
void CodeGenerator::Visit(const PTX::TypedVariableDeclaration<T, S> *declaration)
{
	if constexpr(std::is_same<S, PTX::ParameterSpace>::value)
	{
		const auto names = declaration->GetNames();
		if (names.size() != 1)
		{
			Utils::Logger::LogError("Parameters must only declare a single variable");
		}

		const auto name = names.at(0);
		if (name->GetCount() != 1)
		{
			Utils::Logger::LogError("Parameters must only declare a single variable");
		}

		m_builder.AddParameter(name->GetName(), PTX::BitSize<T::TypeBits>::NumBytes);
	}
}

// Basic Block

bool CodeGenerator::VisitIn(const PTX::BasicBlock *block)
{
	m_builder.CreateBasicBlock(block->GetLabel()->GetName());
	return true;
}

void CodeGenerator::VisitOut(const PTX::BasicBlock *block)
{
	m_builder.CloseBasicBlock();
}

// Statements

bool CodeGenerator::VisitIn(const PTX::InstructionStatement *statement)
{
	InstructionGenerator generator(m_builder);
	statement->Accept(generator);
	return false;
}

}
}
