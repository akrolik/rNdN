#include "Backend/Codegen/CodeGenerator.h"

#include "Backend/Codegen/Generators/InstructionGenerator.h"
#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"

namespace Backend {
namespace Codegen {

// Public API

SASS::Function *CodeGenerator::Generate(const PTX::FunctionDefinition<PTX::VoidType> *function, const PTX::Analysis::RegisterAllocation *registerAllocation, const PTX::Analysis::ParameterSpaceAllocation *parameterAllocation)
{
	// Setup codegen builder

	auto sassFunction = m_builder.CreateFunction(function->GetName());
	m_builder.SetRegisterAllocation(registerAllocation);
	m_builder.SetParameterSpaceAllocation(parameterAllocation);

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

	auto ssyInstruction = new SASS::SSYInstruction("");
	branchBlock->AddInstruction(ssyInstruction);

	auto syncInstruction1 = new SASS::SYNCInstruction();
	auto syncInstruction2 = new SASS::SYNCInstruction();

	// Generate the condition to predicate threads

	PredicateGenerator predicateGenerator(m_builder);
	auto [predicate, negate] = predicateGenerator.Generate(structure->GetPredicate()); 

	if (auto trueStructure = structure->GetTrueBranch())
	{
		SASS::BRAInstruction *branchInstruction = nullptr;

		if (auto falseStructure = structure->GetFalseBranch())
		{
			// If-Else pattern
			// 
			//     SSY L2
			//  @P BRA L1
			//     <False>
			//     SYNC
			// L1:
			//     <True>
			//     SYNC
			// L2:

			// Branch true threads

			branchInstruction = new SASS::BRAInstruction("");
			branchInstruction->SetPredicate(predicate, negate);
			branchBlock->AddInstruction(branchInstruction);

			// Generate false structure

			falseStructure->Accept(*this);

			// Sync false threads

			m_endBlock->AddInstruction(syncInstruction1);
		}
		else
		{
			// Only true branch pattern
			// 
			//     SSY L1
			// @!P SYNC
			//     <True>
			//     SYNC
			// L1:

			// Sync false threads

			syncInstruction1->SetPredicate(predicate, !negate);
			branchBlock->AddInstruction(syncInstruction1);
		}

		// Generate true structure

		trueStructure->Accept(*this);
		auto trueBlock = m_beginBlock;

		// Sync true threads

		m_endBlock->AddInstruction(syncInstruction2);

		// Patch true branch

		if (branchInstruction != nullptr)
		{
			branchInstruction->SetTarget(trueBlock->GetName());
		}
	}
	else if (auto falseStructure = structure->GetFalseBranch())
	{
		// Only false branch pattern
		// 
		//     SSY L1
		//  @P SYNC
		//     <False>
		//     SYNC
		// L1:

		// Sync true threads

		syncInstruction1->SetPredicate(predicate, negate);
		branchBlock->AddInstruction(syncInstruction1);

		// Generate false structure

		falseStructure->Accept(*this);

		// Sync false threads

		m_endBlock->AddInstruction(syncInstruction2);
	}

	// Process next structure

	PTX::Analysis::ConstStructuredGraphVisitor::Visit(structure);

	// Patch SSY reconvergence point

	const auto& reconvergencePoint = m_beginBlock->GetName();

	ssyInstruction->SetTarget(reconvergencePoint);
	m_builder.AddIndirectBranch(syncInstruction1, reconvergencePoint);
	m_builder.AddIndirectBranch(syncInstruction2, reconvergencePoint);

	m_beginBlock = branchBlock;
}

void CodeGenerator::Visit(const PTX::Analysis::ExitStructure *structure)
{
	// Exit pattern
	//
	//     <Block>
	//  @P SYNC
	//     <Next>

	structure->GetBlock()->Accept(*this);
	auto block = m_endBlock;

	// Generate the condition to predicate threads

	auto [structurePredicate, structureNegate] = structure->GetPredicate();

	PredicateGenerator predicateGenerator(m_builder);
	auto [predicate, negate] = predicateGenerator.Generate(structurePredicate); 

	// Predicate threads for exit

	auto syncInstruction = new SASS::SYNCInstruction();
	syncInstruction->SetPredicate(predicate, negate ^ structureNegate);

	m_loopExits.push_back(syncInstruction);
	m_endBlock->AddInstruction(syncInstruction);

	// Process next structure

	PTX::Analysis::ConstStructuredGraphVisitor::Visit(structure);

	m_beginBlock = block;
}

void CodeGenerator::Visit(const PTX::Analysis::LoopStructure *structure)
{
	// Loop pattern
	//
	//     SSY L2
	// L1:
	//     <Body>
	//     BRA L1
	// L2: 
	//     <Next>

	auto ssyInstruction = new SASS::SSYInstruction("");
	m_endBlock->AddInstruction(ssyInstruction);

	// Process loop body

	auto oldExits = m_loopExits;
	m_loopExits.clear();

	structure->GetBody()->Accept(*this);
	m_endBlock->AddInstruction(new SASS::BRAInstruction(m_beginBlock->GetName()));

	auto headerBlock = m_beginBlock;
	auto loopExits = m_loopExits;
	m_loopExits = oldExits;

	// Process next structure

	PTX::Analysis::ConstStructuredGraphVisitor::Visit(structure);

	// Patch header instruction and loop exit branches

	auto reconvergencePoint = m_beginBlock->GetName();
	ssyInstruction->SetTarget(reconvergencePoint);

	for (auto loopExit : loopExits)
	{
		m_builder.AddIndirectBranch(loopExit, reconvergencePoint);
	}

	m_beginBlock = headerBlock;
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
