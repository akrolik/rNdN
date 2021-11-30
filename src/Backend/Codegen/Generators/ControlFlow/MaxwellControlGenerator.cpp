#include "Backend/Codegen/Generators/ControlFlow/MaxwellControlGenerator.h"

#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"

namespace Backend {
namespace Codegen {

void MaxwellControlGenerator::Generate(const PTX::FunctionDefinition<PTX::VoidType> *function)
{
	ControlFlowGenerator::Generate(function);

	// Max stack size required for ELF file

	m_builder.SetCRSStackSize(m_maxStack);
}

// Structure

void MaxwellControlGenerator::Visit(const PTX::Analysis::StructureNode *structure)
{
	// Generate next structure

	if (auto next = structure->GetNext())
	{
		next->Accept(*this);
	}
}

void MaxwellControlGenerator::Visit(const PTX::Analysis::BranchStructure *structure)
{
	// Generate current block, including condition generation

	structure->GetBlock()->Accept(*this);
	auto branchBlock = m_endBlock;

	auto ssyInstruction = new SASS::Maxwell::SSYInstruction("");
	auto syncInstruction1 = new SASS::Maxwell::SYNCInstruction();
	auto syncInstruction2 = new SASS::Maxwell::SYNCInstruction();

	branchBlock->AddInstruction(ssyInstruction);

	m_stackSize += SASS::SSY_STACK_SIZE;
	if (m_stackSize > m_maxStack)
	{
		m_maxStack = m_stackSize;
	}

	// Generate the condition to predicate threads

	PredicateGenerator predicateGenerator(m_builder);
	auto [predicate, negate] = predicateGenerator.Generate(structure->GetPredicate()); 

	if (auto trueStructure = structure->GetTrueBranch())
	{
		SASS::Maxwell::BRAInstruction *branchInstruction = nullptr;

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

			branchInstruction = new SASS::Maxwell::BRAInstruction("");
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

	m_stackSize -= SASS::SSY_STACK_SIZE;

	// Process next structure

	PTX::Analysis::ConstStructuredGraphVisitor::Visit(structure);

	// Patch SSY reconvergence point

	const auto& reconvergencePoint = m_beginBlock->GetName();

	ssyInstruction->SetTarget(reconvergencePoint);
	m_builder.AddIndirectBranch(syncInstruction1, reconvergencePoint);
	m_builder.AddIndirectBranch(syncInstruction2, reconvergencePoint);

	m_beginBlock = branchBlock;
}

void MaxwellControlGenerator::Visit(const PTX::Analysis::ExitStructure *structure)
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

	auto syncInstruction = new SASS::Maxwell::SYNCInstruction();
	syncInstruction->SetPredicate(predicate, negate ^ structureNegate);

	m_loopExits.push_back(syncInstruction);
	m_endBlock->AddInstruction(syncInstruction);

	// Process next structure

	PTX::Analysis::ConstStructuredGraphVisitor::Visit(structure);

	m_beginBlock = block;
}

void MaxwellControlGenerator::Visit(const PTX::Analysis::LoopStructure *structure)
{
	// Loop pattern
	//
	//     SSY L2
	// L1:
	//     <Body>
	//     BRA L1
	// L2: 
	//     <Next>

	auto ssyInstruction = new SASS::Maxwell::SSYInstruction("");
	m_endBlock->AddInstruction(ssyInstruction);

	m_stackSize += SASS::SSY_STACK_SIZE;
	if (m_stackSize > m_maxStack)
	{
		m_maxStack = m_stackSize;
	}

	// Process loop body

	auto oldExits = m_loopExits;
	m_loopExits.clear();

	structure->GetBody()->Accept(*this);
	m_endBlock->AddInstruction(new SASS::Maxwell::BRAInstruction(m_beginBlock->GetName()));

	auto headerBlock = m_beginBlock;
	auto loopExits = m_loopExits;
	m_loopExits = oldExits;

	m_stackSize -= SASS::SSY_STACK_SIZE;

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

void MaxwellControlGenerator::Visit(const PTX::Analysis::SequenceStructure *structure)
{
	// Generate current block

	structure->GetBlock()->Accept(*this);
	auto sequenceBlock = m_endBlock;

	// Process next structure

	PTX::Analysis::ConstStructuredGraphVisitor::Visit(structure);
	m_beginBlock = sequenceBlock;
}

// Basic block

void MaxwellControlGenerator::VisitOut(const PTX::BasicBlock *block)
{
	m_endBlock = m_builder.GetCurrentBlock();

	ControlFlowGenerator::VisitOut(block);
}

}
}
