#include "PTX/Analysis/ControlFlow/ControlFlowBuilder.h"

namespace PTX {
namespace Analysis {

// API

void ControlFlowAccumulator::Analyze(FunctionDefinition<VoidType> *function)
{
	m_graph = new ControlFlowGraph(function);
	function->Accept(*this);
}

// Statements

bool ControlFlowAccumulator::VisitIn(Statement *statement)
{
	// 1. Leader for first statement. May also occur for new blocks throughout
	if (m_currentBlock == nullptr)
	{
		auto label = new Label("_BB" + std::to_string(m_index++));
		m_currentBlock = new BasicBlock(label);
		m_graph->InsertNode(m_currentBlock, label);
	}

	// 2. Leader after branch instruction
	if (auto branchInstruction = dynamic_cast<BranchInstruction *>(statement))
	{
		m_currentBlock->AddStatement(statement);
		m_statementMap[statement] = m_currentBlock;

		m_currentBlock = nullptr;
		return false;
	}

	// 3. False branch if predicated instruction
	if (auto predInstruction = dynamic_cast<PredicatedInstruction *>(statement))
	{
		if (auto [predicate, negate] = predInstruction->GetPredicate(); predicate != nullptr)
		{
			auto label = new Label("_PRED" + std::to_string(m_index));
			auto labelEnd = new Label("_PEND" + std::to_string(m_index));

			// Insert branch into existing block

			auto branchStatement = new PTX::BranchInstruction(labelEnd, predicate, !negate);
			m_currentBlock->AddStatement(branchStatement);
			m_statementMap[branchStatement] = m_currentBlock;

			// Current instruction goes in its own special block

			auto inBlock = m_currentBlock;
			auto predBlock = new BasicBlock(label);

			m_currentBlock = predBlock;
			m_graph->InsertNode(m_currentBlock, label);

			// Remove predicate (now part of the block)

			predInstruction->SetPredicate(nullptr);

			m_currentBlock->AddStatement(predInstruction);
			m_statementMap[predInstruction] = m_currentBlock;

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
	m_statementMap[statement] = m_currentBlock;

	return false;
}

bool ControlFlowAccumulator::VisitIn(LabelStatement *statement)
{
	// 4. Leader for branch target
	auto label = statement->GetLabel();
	auto labelBlock = new BasicBlock(label);
	m_graph->InsertNode(labelBlock, label);
	m_labelMap[label] = labelBlock;

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

void ControlFlowBuilder::Analyze(FunctionDefinition<VoidType> *function)
{
	ControlFlowAccumulator accumulator;
	accumulator.Analyze(function);

	m_graph = accumulator.GetGraph();
	m_labelMap = accumulator.GetLabelMap();
	m_statementMap = accumulator.GetStatementMap();

	function->Accept(*this);
}

// Statements

bool ControlFlowBuilder::VisitIn(Statement *statement)
{
	// Insert incoming edges if needed

	auto statementBlock = m_statementMap.at(statement);
	if (m_previousBlock != nullptr && m_previousBlock != statementBlock)
	{
		m_graph->InsertEdge(m_previousBlock, statementBlock);
	}
	m_previousBlock = statementBlock;
	return false;
}

bool ControlFlowBuilder::VisitIn(InstructionStatement *statement)
{
	// Insert incoming edges

	HierarchicalVisitor::VisitIn(statement);

	// Insert outgoing edges for branching (label target)

	if (auto branchInstruction = dynamic_cast<BranchInstruction *>(statement))
	{
		auto statementBlock = m_statementMap.at(statement);
		m_graph->InsertEdge(statementBlock, m_labelMap.at(branchInstruction->GetLabel()));

		// Predicated branches have an edge to the next block

		if (branchInstruction->HasPredicate())
		{
			m_previousBlock = statementBlock;
		}
	}
	// Predicate instructions cause internal control flow
	else if (auto predInstruction = dynamic_cast<PredicatedInstruction *>(statement))
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
