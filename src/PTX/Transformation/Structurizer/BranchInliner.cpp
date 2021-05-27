#include "PTX/Transformation/Structurizer/BranchInliner.h"

#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraphPrinter.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace PTX {
namespace Transformation {

Analysis::StructureNode *BranchInliner::Optimize(FunctionDefinition<VoidType> *function)
{
	auto timeStructurize_start = Utils::Chrono::Start("Predicated structurizer '" + function->GetName() + "'");

	m_sequenceNode = nullptr;
	m_nextNode = nullptr;
	m_nextBlock = nullptr;

	function->GetStructuredGraph()->Accept(*this);

	Utils::Chrono::End(timeStructurize_start);

	if (Utils::Options::IsBackend_PrintStructured())
	{
		Utils::Logger::LogInfo("Predicated structured control-flow graph: " + function->GetName());

		auto structureString = Analysis::StructuredGraphPrinter::PrettyString(function->GetName(), m_nextNode);
		Utils::Logger::LogInfo(structureString, 0, true, Utils::Logger::NoPrefix);
	}

	return m_nextNode;
}

void BranchInliner::Visit(Analysis::BranchStructure *structure)
{
	// Reset predicate

	auto oldPredicate = m_predicate;
	auto predicate = structure->GetPredicate();

	m_predicated = true;
	m_predicate = predicate;

	// Check both true/false structures are predicatable
	//   - No nested structures
	//   - Each branch is a single basic block
	//   - Predicate not used in instructions
	//   - No predicated instructions

	Analysis::SequenceStructure *trueSequence = nullptr;
	Analysis::SequenceStructure *falseSequence = nullptr;

	Analysis::StructureNode *trueStructure = nullptr;
	Analysis::StructureNode *falseStructure = nullptr;

	if (auto trueBranch = structure->GetTrueBranch())
	{
		m_sequenceNode = nullptr;
		m_nextNode = nullptr;
		m_nextBlock = nullptr;

		trueBranch->Accept(*this);
		trueSequence = m_sequenceNode;
		trueStructure = m_nextNode;
	}

	if (auto falseBranch = structure->GetFalseBranch())
	{
		m_sequenceNode = nullptr;
		m_nextNode = nullptr;
		m_nextBlock = nullptr;

		falseBranch->Accept(*this);
		falseSequence = m_sequenceNode;
		falseStructure = m_nextNode;
	}

	auto predicated = m_predicated;
	auto block = structure->GetBlock();

	// Construct structure that follows

	Analysis::StructuredGraphVisitor::Visit(structure);

	// If legal, merge the true/false structures

	if (predicated)
	{
		// Construct an inlined block with the true/false branches

		std::vector<Statement *> statements;
		for (auto& statement : structure->GetBlock()->GetStatements())
		{
			statements.push_back(statement);
		}

		// Add true statements (all may predicated)

		if (trueSequence != nullptr)
		{
			for (auto& instruction : trueSequence->GetBlock()->GetStatements())
			{
				if (auto predicatedInstruction = dynamic_cast<PredicatedInstruction *>(instruction))
				{
					predicatedInstruction->SetPredicate(predicate, false);
					statements.push_back(predicatedInstruction);
				}
			}
		}

		// Add false statements (all may be predicated)

		if (falseSequence != nullptr)
		{
			for (auto& instruction : falseSequence->GetBlock()->GetStatements())
			{
				if (auto predicatedInstruction = dynamic_cast<PredicatedInstruction *>(instruction))
				{
					predicatedInstruction->SetPredicate(predicate, true);
					statements.push_back(predicatedInstruction);
				}
			}
		}

		// Finally, continue with the next block statements

		if (m_nextBlock != nullptr)
		{
			auto& nextStatements = m_nextBlock->GetStatements();
			statements.insert(std::end(statements), std::begin(nextStatements), std::end(nextStatements));
			m_nextBlock->SetStatements(statements);
			m_nextBlock->SetLabel(block->GetLabel());

			// Propagate m_nextBlock, m_nextNode
		}
		else
		{
			auto newBlock = new BasicBlock(block->GetLabel());
			newBlock->SetStatements(statements);

			m_nextNode = new Analysis::SequenceStructure(newBlock, m_nextNode);
			m_nextBlock = newBlock;
		}
	}
	else
	{
		m_nextNode = new Analysis::BranchStructure(block, predicate, trueStructure, falseStructure, m_nextNode);
		m_nextBlock = block;
	}

	m_sequenceNode = nullptr;

	m_predicate = oldPredicate;
	m_predicated = false;
}

void BranchInliner::Visit(const InstructionStatement *instruction)
{
	// Non-predicated instructions cannot be inlined

	m_predicated = false;
}

void BranchInliner::Visit(const PredicatedInstruction *instruction)
{
	// If there is already a predicate, cannot be inlined

	if (instruction->GetPredicate().first != nullptr)
	{
		m_predicated = false;
		return;
	}

	// Some instructions must be handled separately

	instruction->Accept(static_cast<ConstInstructionVisitor&>(*this));

	// Verify all operands are legal in predicated branch

	for (auto& operand : instruction->GetOperands())
	{
		operand->Accept(static_cast<ConstOperandVisitor&>(*this));
	}
}

void BranchInliner::Visit(const _RemainderInstruction *instruction)
{
	m_predicated = false;
}

void BranchInliner::Visit(const _DivideInstruction *instruction)
{
	m_predicated = false;
}

void BranchInliner::Visit(const _Register *reg)
{
	reg->Dispatch(*this);
}

template<class T>
void BranchInliner::Visit(const Register<T> *reg)
{
	// Check that the register is not used in any instruction (for simplicity, use/definition)

	if constexpr(std::is_same<T, PredicateType>::value)
	{
		if (m_predicate != nullptr && reg->GetName() == m_predicate->GetName())
		{
			m_predicated = false;
		}
	}
}

void BranchInliner::Visit(Analysis::ExitStructure *structure)
{
	// Traverse the next structure

	Analysis::StructuredGraphVisitor::Visit(structure);

	// Create new exit structure

	auto [predicate, negate] = structure->GetPredicate();
	auto block = structure->GetBlock();

	auto newNode = new Analysis::ExitStructure(block, predicate, negate, m_nextNode);
	m_exitStructures.insert(newNode);

	m_nextNode = newNode;
	m_sequenceNode = nullptr;
	m_nextBlock = block;

	if (m_latchStructure == structure)
	{
		m_latchStructure = m_nextNode;
	}

	// Cannot be predicated

	m_predicated = false;
}

void BranchInliner::Visit(Analysis::LoopStructure *structure)
{
	// Traverse the next structure

	Analysis::StructuredGraphVisitor::Visit(structure);
	auto next = m_nextNode;

	// Traverse loop body

	auto oldExits = m_exitStructures;
	auto oldLatch = m_latchStructure;

	m_exitStructures.clear();
	m_latchStructure = structure->GetLatch();

	structure->GetBody()->Accept(*this);
	auto body = m_nextNode;

	// Create new loop structure

	m_nextNode = new Analysis::LoopStructure(body, m_exitStructures, m_latchStructure, next);
	m_sequenceNode = nullptr;
	m_nextBlock = nullptr;

	// Cannot be predicated

	m_predicated = false;

	// Restore old structures

	m_exitStructures = oldExits;
	m_latchStructure = oldLatch;
}

void BranchInliner::Visit(Analysis::SequenceStructure *structure)
{
	// If this is a sequence structure, check all instructions are legal

	auto& statements = structure->GetBlock()->GetStatements();
	if (statements.size() <= Utils::Options::GetBackend_InlineBranchThreshold())
	{
		for (auto& instruction : structure->GetBlock()->GetStatements())
		{
			instruction->Accept(*this);
		}
	}
	else
	{
		m_predicated = false;
	}

	Analysis::StructuredGraphVisitor::Visit(structure);

	// Create new sequence structure

	auto block = structure->GetBlock();

	m_nextNode = m_sequenceNode = new Analysis::SequenceStructure(block, m_nextNode);
	m_nextBlock = block;

	if (m_latchStructure == structure)
	{
		m_latchStructure = m_nextNode;
	}
}

void BranchInliner::Visit(Analysis::StructureNode *structure)
{
	// Traverse next structure node

	m_nextNode = nullptr;
	m_sequenceNode = nullptr;
	m_nextBlock = nullptr;

	if (auto next = structure->GetNext())
	{
		next->Accept(*this);
		m_predicated = false;
	}
}

}
}
