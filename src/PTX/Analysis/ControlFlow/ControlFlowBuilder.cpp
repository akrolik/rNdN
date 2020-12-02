#include "PTX/Analysis/ControlFlow/ControlFlowBuilder.h"

namespace PTX {
namespace Analysis {

// API

void ControlFlowAccumulator::Analyze(const FunctionDefinition<VoidType> *function)
{
	m_graph = new ControlFlowGraph();
	function->Accept(*this);
}

// Statements

bool ControlFlowAccumulator::VisitIn(const Statement *statement)
{
	if (m_currentBlock == nullptr)
	{
		m_currentBlock = new BasicBlock("_start");
		m_graph->InsertNode(m_currentBlock);
	}
	m_currentBlock->AddStatement(statement);
	m_blockMap[statement] = m_currentBlock;
	return false;
}

bool ControlFlowAccumulator::VisitIn(const Label *label)
{
	auto previousBlock = m_currentBlock;
	m_currentBlock = new BasicBlock(label->GetName());
	m_graph->InsertNode(m_currentBlock);
	m_blockMap[label] = m_currentBlock;
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
	m_previousBlock = m_blockMap.at(statement);
	return false;
}

bool ControlFlowBuilder::VisitIn(const InstructionStatement *statement)
{
	auto statementBlock = m_blockMap.at(statement);
	if (auto branchInstruction = dynamic_cast<const BranchInstruction *>(statement))
	{
		m_graph->InsertEdge(statementBlock, m_blockMap.at(branchInstruction->GetLabel()));
		if (!branchInstruction->HasPredicate())
		{
			m_previousBlock = nullptr;
			return false;
		}
	}
	m_previousBlock = statementBlock;
	return false;
}

bool ControlFlowBuilder::VisitIn(const Label *label)
{
	if (m_previousBlock != nullptr)
	{
		m_graph->InsertEdge(m_previousBlock, m_blockMap.at(label));
	}
	return false;
}

}
}
