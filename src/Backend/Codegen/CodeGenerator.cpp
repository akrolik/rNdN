#include "Backend/Codegen/CodeGenerator.h"

#include "Backend/Codegen/Generators/InstructionGenerator.h"

namespace Backend {
namespace Codegen {

// Public API

SASS::Function *CodeGenerator::Generate(const PTX::FunctionDefinition<PTX::VoidType> *function, const PTX::Analysis::RegisterAllocation *registerAllocation, const PTX::Analysis::LocalSpaceAllocation *spaceAllocation)
{
	// Setup codegen builder

	auto sassFunction = m_builder.CreateFunction(function->GetName());
	m_builder.SetRegisterAllocation(registerAllocation);
	m_builder.SetLocalSpaceAllocation(spaceAllocation);

	// Traverse function

	function->Accept(*this);

	// Close function and return

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
		m_builder.AddParameter(PTX::BitSize<T::TypeBits>::NumBytes);
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
	// Clear all allocated temporary registers

	m_builder.ClearTemporaryRegisters();

	// Generate instruction

	InstructionGenerator generator(m_builder);
	statement->Accept(generator);
	return false;
}

}
}
