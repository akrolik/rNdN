#include "Backend/Codegen/CodeGenerator.h"

#include "Backend/Codegen/Generators/InstructionGenerator.h"
#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"

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

bool CodeGenerator::VisitIn(const PTX::FunctionDefinition<PTX::VoidType> *function)
{
	// Construct parameters

	for (const auto& parameter : function->GetParameters())
	{
		parameter->Accept(static_cast<ConstHierarchicalVisitor&>(*this));
	}

	// Construct basic blocks

	function->GetStructuredGraph()->Accept(*this);

	return false;
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

// Structure

void CodeGenerator::Visit(const PTX::Analysis::StructureNode *structure)
{
	// Generate next structure

	if (auto next = structure->GetNext())
	{
		next->Accept(*this);
	}
}

void CodeGenerator::Visit(const PTX::Analysis::BranchStructure *structure)
{
	// Generate current block, including condition generation

	structure->GetBlock()->Accept(*this);
	auto branchBlock = m_endBlock;

	// Generate the condition to predicate threads

	PredicateGenerator predicateGenerator(m_builder);
	auto [predicate, negate] = predicateGenerator.Generate(structure->GetPredicate()); 

	SASS::PredicatedInstruction *branchInstruction = nullptr;

	if (auto trueStructure = structure->GetTrueBranch())
	{
		if (auto falseStructure = structure->GetFalseBranch())
		{
			// Generate false structure

			falseStructure->Accept(*this);

			// Sync false threads

			m_endBlock->AddInstruction(new SASS::SYNCInstruction());
		}
		else
		{
			// Sync false threads

			branchInstruction = new SASS::SYNCInstruction();
			branchInstruction->SetPredicate(predicate, !negate);
		}

		// Generate true structure

		trueStructure->Accept(*this);

		// Sync true threads

		m_endBlock->AddInstruction(new SASS::SYNCInstruction());

		if (auto falseStructure = structure->GetFalseBranch())
		{
			// Branch true threads

			branchInstruction = new SASS::BRAInstruction(m_beginBlock->GetName());
			branchInstruction->SetPredicate(predicate, negate);
		}
	}
	else if (auto falseStructure = structure->GetFalseBranch())
	{
		// Sync true threads

		branchInstruction = new SASS::SYNCInstruction();
		branchInstruction->SetPredicate(predicate, negate);

		// Generate false structure

		falseStructure->Accept(*this);

		// Sync false threads

		m_endBlock->AddInstruction(new SASS::SYNCInstruction());
	}

	// Process next structure

	PTX::Analysis::ConstStructuredGraphVisitor::Visit(structure);
	branchBlock->AddInstruction(new SASS::SSYInstruction(m_beginBlock->GetName()));
	branchBlock->AddInstruction(branchInstruction);

	m_beginBlock = branchBlock;
}

void CodeGenerator::Visit(const PTX::Analysis::ExitStructure *structure)
{
	//TODO: Codegen exit structure
}

void CodeGenerator::Visit(const PTX::Analysis::LoopStructure *structure)
{
	//TODO: Codegen loop structure
}

void CodeGenerator::Visit(const PTX::Analysis::SequenceStructure *structure)
{
	// Generate current block

	structure->GetBlock()->Accept(*this);
	auto sequenceBlock = m_endBlock;

	// Process next structure

	PTX::Analysis::ConstStructuredGraphVisitor::Visit(structure);
	m_beginBlock = sequenceBlock;
}

// Basic Block

bool CodeGenerator::VisitIn(const PTX::BasicBlock *block)
{
	m_endBlock = m_builder.CreateBasicBlock(block->GetLabel()->GetName());
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
