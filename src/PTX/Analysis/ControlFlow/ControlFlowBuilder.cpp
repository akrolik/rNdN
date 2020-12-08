#include "PTX/Analysis/ControlFlow/ControlFlowBuilder.h"

namespace PTX {
namespace Analysis {

// API

void ControlFlowAccumulator::Analyze(const FunctionDefinition<VoidType> *function)
{
	m_graph = new ControlFlowGraph(function);
	function->Accept(*this);
}

// Statements

bool ControlFlowAccumulator::VisitIn(const Statement *statement)
{
	// 1. Leader for first statement. May also occur for new blocks throughout
	if (m_currentBlock == nullptr)
	{
		auto label = new Label("_BB" + std::to_string(m_index++));
		m_currentBlock = new BasicBlock(label);
		m_graph->InsertNode(m_currentBlock, label);
	}

	// 2. Leader after branch instruction
	if (auto branchInstruction = dynamic_cast<const BranchInstruction *>(statement))
	{
		m_currentBlock->AddStatement(statement);
		m_blockMap[statement] = m_currentBlock;

		m_currentBlock = nullptr;
		return false;
	}

	// 3. False branch if predicated instruction
	if (auto predInstruction = dynamic_cast<const PredicatedInstruction *>(statement))
	{
		if (auto [predicate, negate] = predInstruction->GetPredicate(); predicate != nullptr)
		{
			auto label = new Label("_PRED" + std::to_string(m_index));
			auto labelEnd = new Label("_PEND" + std::to_string(m_index));

			// Insert branch into existing block

			auto branchStatement = new PTX::BranchInstruction(labelEnd, predicate, !negate);
			m_currentBlock->AddStatement(branchStatement);
			m_blockMap[branchStatement] = m_currentBlock;

			// Current instruction goes in its own special block

			auto inBlock = m_currentBlock;
			auto predBlock = new BasicBlock(label);

			m_currentBlock = predBlock;
			m_graph->InsertNode(m_currentBlock, label);

			//TODO: Remove predicate (clone)
			m_currentBlock->AddStatement(predInstruction);
			m_blockMap[predInstruction] = m_currentBlock;

			// Converge branches

			m_currentBlock = new BasicBlock(labelEnd);
			m_graph->InsertNode(m_currentBlock, labelEnd);

			// Insert edges

			m_graph->InsertEdge(inBlock, predBlock);
			m_graph->InsertEdge(inBlock, m_currentBlock);
			m_graph->InsertEdge(predBlock, m_currentBlock);

			m_index++;
			return false;
		}
	}

	m_currentBlock->AddStatement(statement);
	m_blockMap[statement] = m_currentBlock;

	return false;
}

bool ControlFlowAccumulator::VisitIn(const Label *label)
{
	// 4. Leader for branch target
	auto labelBlock = new BasicBlock(label);
	m_graph->InsertNode(labelBlock, label);
	m_blockMap[label] = labelBlock;

	// Empty block occurs if the previous statement was a predicated (non-branch) instruction
	if (m_currentBlock != nullptr)
	{
		if (m_currentBlock->GetSize() == 0)
		{
			m_graph->InsertEdge(m_currentBlock, labelBlock);
		}
	}
	m_currentBlock = labelBlock;

	return false;
}

// Builder API

void ControlFlowBuilder::Analyze(const FunctionDefinition<VoidType> *function)
{
	ControlFlowAccumulator accumulator;
	accumulator.Analyze(function);

	m_graph = accumulator.GetGraph();
	m_blockMap = accumulator.GetBlockMap();

	function->Accept(*this);
}

// Statements

bool ControlFlowBuilder::VisitIn(const Statement *statement)
{
	// Insert incoming edges if needed

	auto statementBlock = m_blockMap.at(statement);
	if (m_previousBlock != nullptr && m_previousBlock != statementBlock)
	{
		m_graph->InsertEdge(m_previousBlock, statementBlock);
	}
	m_previousBlock = statementBlock;
	return false;
}

bool ControlFlowBuilder::VisitIn(const InstructionStatement *statement)
{
	// Insert incoming edges

	ConstHierarchicalVisitor::VisitIn(statement);

	// Insert outgoing edges for branching (label target)

	if (auto branchInstruction = dynamic_cast<const BranchInstruction *>(statement))
	{
		auto statementBlock = m_blockMap.at(statement);
		m_graph->InsertEdge(statementBlock, m_blockMap.at(branchInstruction->GetLabel()));

		// Predicated branches have an edge to the next block

		if (branchInstruction->HasPredicate())
		{
			m_previousBlock = statementBlock;
		}
	}
	// Predicate instructions cause internal control flow
	else if (auto predInstruction = dynamic_cast<const PredicatedInstruction *>(statement))
	{
		if (predInstruction->HasPredicate())
		{
			m_previousBlock = nullptr;
		}
	}
	return false;
}

}
}
