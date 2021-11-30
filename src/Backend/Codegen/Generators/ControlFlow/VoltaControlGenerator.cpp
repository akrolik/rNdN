#include "Backend/Codegen/Generators/ControlFlow/VoltaControlGenerator.h"

#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"

namespace Backend {
namespace Codegen {

void VoltaControlGenerator::Visit(const PTX::Analysis::StructureNode *structure)
{
	// Generate next structure

	if (auto next = structure->GetNext())
	{
		next->Accept(*this);
	}
}

void VoltaControlGenerator::Visit(const PTX::Analysis::BranchStructure *structure)
{
	// Generate current block, including condition generation

	structure->GetBlock()->Accept(*this);
	auto branchBlock = m_endBlock;

	// Associate reconvergence barrier with the divergence point

	auto ssyInstruction = new SASS::Volta::BSSYInstruction("", m_stackDepth);
	branchBlock->AddInstruction(ssyInstruction);

	// Verify free barriers remain

	if (m_stackDepth > SASS::Volta::MAX_SSY_STACK_DEPTH)
	{
		Utils::Logger::LogError("Stack depth " + std::to_string(m_stackDepth) + " exceeded maximum (" + std::to_string(SASS::Volta::MAX_SSY_STACK_DEPTH) + ")");
	}
	m_stackDepth++;

	// Generate the condition to predicate threads

	PredicateGenerator predicateGenerator(m_builder);
	auto [predicate, negate] = predicateGenerator.Generate(structure->GetPredicate()); 

	// Generate if-else structure

	auto branchEndInstruction = new SASS::Volta::BRAInstruction("");
	if (auto trueStructure = structure->GetTrueBranch())
	{
		auto branchElseInstruction = new SASS::Volta::BRAInstruction("");
		if (auto falseStructure = structure->GetFalseBranch())
		{
			// If-Else pattern
			// 
			//     BSSY B, L2
			//  @P BRA L1
			//     <False>
			//     BRA L2
			// L1:
			//     <True>
			// L2:
			//     BSYNC B
			// L3:

			// Branch true threads

			branchElseInstruction->SetPredicate(predicate, negate);
			branchBlock->AddInstruction(branchElseInstruction);

			// Generate false structure

			falseStructure->Accept(*this);

			// Branch false threads to end

			m_endBlock->AddInstruction(branchEndInstruction);
		}
		else
		{
			// Only true branch pattern
			// 
			//     BSSY B, L2
			// @!P BRA L1
			//     <True>
			// L1:
			//     BSYNC B
			// L2:

			// Branch false threads to end

			branchEndInstruction->SetPredicate(predicate, !negate);
			branchBlock->AddInstruction(branchEndInstruction);
		}

		// Generate true structure

		trueStructure->Accept(*this);
		auto trueBlock = m_beginBlock;

		// Patch true branch

		if (branchElseInstruction != nullptr)
		{
			branchElseInstruction->SetTarget(trueBlock->GetName());
		}
	}
	else if (auto falseStructure = structure->GetFalseBranch())
	{
		// Only false branch pattern
		// 
		//     BSSY B, L2
		//  @P BRA L1
		//     <False>
		// L1:
		//     BSYNC B
		// L2:

		// Branch true threads to end

		branchEndInstruction->SetPredicate(predicate, negate);
		branchBlock->AddInstruction(branchEndInstruction);

		// Generate false structure

		falseStructure->Accept(*this);
	}

	// Create barrier synchronization point

	m_stackDepth--;

	auto syncBlock = m_builder.CreateBasicBlock(m_builder.UniqueIdentifier(branchBlock->GetName() + "_SYNC"));
	syncBlock->AddInstruction(new SASS::Volta::BSYNCInstruction(m_stackDepth));
	m_builder.CloseBasicBlock();

	// Patch end branch

	branchEndInstruction->SetTarget(syncBlock->GetName());

	// Process next structure

	PTX::Analysis::ConstStructuredGraphVisitor::Visit(structure);

	// Patch SSY reconvergence point

	const auto& reconvergencePoint = m_beginBlock->GetName();
	ssyInstruction->SetTarget(reconvergencePoint);

	m_beginBlock = branchBlock;
}

void VoltaControlGenerator::Visit(const PTX::Analysis::ExitStructure *structure)
{
	// Exit pattern
	//
	//     <Block>
	//  @P BRA <empty> (will be patched)
	//     <Next>

	structure->GetBlock()->Accept(*this);
	auto block = m_endBlock;

	// Generate the condition to predicate threads

	auto [structurePredicate, structureNegate] = structure->GetPredicate();

	PredicateGenerator predicateGenerator(m_builder);
	auto [predicate, negate] = predicateGenerator.Generate(structurePredicate); 

	// Predicate threads for exit

	auto branchInstruction = new SASS::Volta::BRAInstruction("");
	branchInstruction->SetPredicate(predicate, negate ^ structureNegate);

	m_loopExits.push_back(branchInstruction);
	m_endBlock->AddInstruction(branchInstruction);

	// Process next structure

	PTX::Analysis::ConstStructuredGraphVisitor::Visit(structure);

	m_beginBlock = block;
}

void VoltaControlGenerator::Visit(const PTX::Analysis::LoopStructure *structure)
{
	// Loop pattern
	//
	//     BSSY B, L3
	// L1:
	//     <Body>
	//     BRA L1
	// L2:
	//     BSYNC B
	// L3: 
	//     <Next>

	// Check barrier resource is available

	if (m_stackDepth > SASS::Volta::MAX_SSY_STACK_DEPTH)
	{
		Utils::Logger::LogError("Stack depth " + std::to_string(m_stackDepth) + " exceeded maximum (" + std::to_string(SASS::Volta::MAX_SSY_STACK_DEPTH) + ")");
	}

	// Setup divergence barrier

	auto ssyInstruction = new SASS::Volta::BSSYInstruction("", m_stackDepth);
	m_endBlock->AddInstruction(ssyInstruction);

	m_stackDepth++;

	// Process loop body

	auto oldExits = m_loopExits;
	m_loopExits.clear();

	structure->GetBody()->Accept(*this);
	m_endBlock->AddInstruction(new SASS::Volta::BRAInstruction(m_beginBlock->GetName()));

	auto headerBlock = m_beginBlock;
	auto loopExits = m_loopExits;
	m_loopExits = oldExits;

	// Create barrier synchronization point

	m_stackDepth--;

	auto syncBlock = m_builder.CreateBasicBlock(m_builder.UniqueIdentifier(headerBlock->GetName() + "_SYNC"));
	syncBlock->AddInstruction(new SASS::Volta::BSYNCInstruction(m_stackDepth));
	m_builder.CloseBasicBlock();

	// Process next structure

	PTX::Analysis::ConstStructuredGraphVisitor::Visit(structure);

	// Patch header instruction and loop exit branches

	auto reconvergencePoint = m_beginBlock->GetName();
	ssyInstruction->SetTarget(reconvergencePoint);

	for (auto loopExit : loopExits)
	{
		loopExit->SetTarget(syncBlock->GetName());
	}

	m_beginBlock = headerBlock;
}

void VoltaControlGenerator::Visit(const PTX::Analysis::SequenceStructure *structure)
{
	// Generate current block

	structure->GetBlock()->Accept(*this);
	auto sequenceBlock = m_endBlock;

	// Process next structure

	PTX::Analysis::ConstStructuredGraphVisitor::Visit(structure);
	m_beginBlock = sequenceBlock;
}
 
// Basic block

void VoltaControlGenerator::VisitOut(const PTX::BasicBlock *block)
{
	m_endBlock = m_builder.GetCurrentBlock();

	ControlFlowGenerator::VisitOut(block);
}

}
}
