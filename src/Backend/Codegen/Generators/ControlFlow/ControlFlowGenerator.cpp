#include "Backend/Codegen/Generators/ControlFlow/ControlFlowGenerator.h"

#include "Backend/Codegen/Generators/InstructionGenerator.h"

namespace Backend {
namespace Codegen {

// Public API

void ControlFlowGenerator::Generate(const PTX::FunctionDefinition<PTX::VoidType> *function)
{
	// Traverse function

	for (const auto& parameter : function->GetParameters())
	{
		parameter->Accept(static_cast<ConstHierarchicalVisitor&>(*this));
	}

	// Construct basic blocks

	function->GetStructuredGraph()->Accept(*this);
}

// Declarations

bool ControlFlowGenerator::VisitIn(const PTX::VariableDeclaration *declaration)
{
	declaration->Accept(static_cast<ConstDeclarationVisitor&>(*this));
	return false;
}

void ControlFlowGenerator::Visit(const PTX::_TypedVariableDeclaration *declaration)
{
	declaration->Dispatch(*this);
}

template<class T, class S>
void ControlFlowGenerator::Visit(const PTX::TypedVariableDeclaration<T, S> *declaration)
{
	for (const auto& name : declaration->GetNames())
	{
		for (auto i = 0u; i < name->GetCount(); ++i)
		{
			const auto string = name->GetName(i);
			const auto dataSize = PTX::BitSize<T::TypeBits>::NumBytes;

			if constexpr(std::is_same<S, PTX::ParameterSpace>::value)
			{
				// Add each parameter declaration to the parameter constant space

				m_builder.AddParameter(dataSize);
			}
			else if constexpr(std::is_same<S, PTX::SharedSpace>::value)
			{
				// Add each shared declaration to the function

				if constexpr(PTX::is_array_type<T>::value)
				{
					// Array sizes, only possible for shared spaces (not parameters)

					m_builder.AddSharedVariable(string, T::ElementCount * dataSize, dataSize);
				}
				else
				{
					m_builder.AddSharedVariable(string, dataSize, dataSize);
				}
			}
		}
	}
}

// Basic Block

bool ControlFlowGenerator::VisitIn(const PTX::BasicBlock *block)
{
	m_builder.CreateBasicBlock(block->GetLabel()->GetName());
	return true;
}

void ControlFlowGenerator::VisitOut(const PTX::BasicBlock *block)
{
	m_builder.CloseBasicBlock();
}

// Statements

bool ControlFlowGenerator::VisitIn(const PTX::InstructionStatement *statement)
{
	// Clear all allocated temporary registers

	m_builder.ClearTemporaryRegisters();

	// Generate instruction

	InstructionGenerator generator(m_builder);
	generator.Generate(statement);
	return false;
}

}
}
