#include "PTX/Analysis/ControlFlow/ControlFlowBuilder.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace PTX {
namespace Analysis {

// API

ControlFlowGraph *ControlFlowAccumulator::Analyze(FunctionDefinition<VoidType> *function)
{
	m_graph = new ControlFlowGraph(function);
	function->Accept(*this);
	return m_graph;
}

// Statements

BasicBlock *ControlFlowAccumulator::CreateBlock()
{
	return CreateBlock("_BB" + std::to_string(m_index++));
}

BasicBlock *ControlFlowAccumulator::CreateBlock(const std::string& name)
{
	auto label = new Label(name);
	auto block = new BasicBlock(label);
	m_graph->InsertNode(block);

	// First basic block is an entry node

	if (m_entry)
	{
		m_graph->SetEntryNode(block);
		m_entry = false;
	}

	return block;
}

bool ControlFlowAccumulator::VisitIn(Statement *statement)
{
	// 1. Leader for first statement. May also occur for new blocks throughout
	if (m_currentBlock == nullptr)
	{
		m_currentBlock = CreateBlock();
	}

	m_currentBlock->AddStatement(statement);
	m_statementMap[statement] = m_currentBlock;

	return false;
}

bool ControlFlowAccumulator::VisitIn(PredicatedInstruction *instruction)
{
	// 2. Leader after branch/return instruction
	if (dynamic_cast<BranchInstruction *>(instruction) || dynamic_cast<ReturnInstruction *>(instruction))
	{
		// Add instruction, then end the current block

		HierarchicalVisitor::VisitIn(instruction);

		if (dynamic_cast<ReturnInstruction *>(instruction))
		{
			m_graph->AddExitNode(m_currentBlock);
		}
		m_currentBlock = nullptr;
	}

	// 3. False branch if predicated instruction
	else if (auto [predicate, negate] = instruction->GetPredicate(); predicate != nullptr)
	{
		if (m_currentBlock == nullptr)
		{
			m_currentBlock = CreateBlock();
		}

		auto label = new Label("_PRED" + std::to_string(m_index));
		auto labelEnd = new Label("_PEND" + std::to_string(m_index));

		// Insert branches into existing block

		auto branchInstruction1 = new BranchInstruction(labelEnd, predicate, !negate);
		auto branchInstruction2 = new BranchInstruction(label);

		auto inBlock = m_currentBlock;
		inBlock->AddStatement(branchInstruction1);
		inBlock->AddStatement(branchInstruction2);

		// Current instruction goes in its own special block

		auto predBlock = new BasicBlock(label);
		m_graph->InsertNode(predBlock);

		predBlock->AddStatement(instruction);
		m_statementMap[instruction] = predBlock;

		auto branchInstruction3 = new BranchInstruction(labelEnd);
		predBlock->AddStatement(branchInstruction3);

		// Converge branches

		m_currentBlock = new BasicBlock(labelEnd);
		m_graph->InsertNode(m_currentBlock);

		// Insert edges

		m_graph->InsertEdge(inBlock, predBlock);
		m_graph->InsertEdge(inBlock, m_currentBlock, predicate, !negate);
		m_graph->InsertEdge(predBlock, m_currentBlock);

		m_index++;
	}
	else
	{
		// Add instruction to block

		HierarchicalVisitor::VisitIn(instruction);
	}

	return false;
}

bool ControlFlowAccumulator::VisitIn(LabelStatement *statement)
{
	// 4. Leader for branch target
	auto label = statement->GetLabel();
	auto labelBlock = new BasicBlock(label);
	m_graph->InsertNode(labelBlock);
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

	// Program starts with a label, entry node special case

	if (m_entry)
	{
		m_graph->SetEntryNode(m_currentBlock);
		m_entry = false;
	}

	return false;
}

// Builder API

ControlFlowGraph *ControlFlowBuilder::Analyze(FunctionDefinition<VoidType> *function)
{
	auto timeBuild_start = Utils::Chrono::Start("Control-flow builder '" + function->GetName() + "'");

	// Accumulate statements into basic blocks

	ControlFlowAccumulator accumulator;
	m_graph = accumulator.Analyze(function);

	// Add edges between blocks

	m_labelMap = accumulator.GetLabelMap();
	m_statementMap = accumulator.GetStatementMap();

	function->Accept(*this);

	Utils::Chrono::End(timeBuild_start);

	// Debug printing

	if (Utils::Options::IsBackend_PrintCFG())
	{
		Utils::Logger::LogInfo("Control-flow graph: " + function->GetName());
		Utils::Logger::LogInfo(m_graph->ToDOTString(), 0, true, Utils::Logger::NoPrefix);
	}

	return m_graph;
}

// Statements

bool ControlFlowBuilder::VisitIn(Statement *statement)
{
	// Insert incoming edges if needed (fallthrough)

	auto statementBlock = m_statementMap.at(statement);
	if (m_previousBlock != nullptr && m_previousBlock != statementBlock)
	{
		m_previousBlock->AddStatement(new BranchInstruction(statementBlock->GetLabel()));
		m_graph->InsertEdge(m_previousBlock, statementBlock);
	}
	m_previousBlock = statementBlock;
	return false;
}

bool ControlFlowBuilder::VisitIn(PredicatedInstruction *instruction)
{
	if (auto branchInstruction = dynamic_cast<BranchInstruction *>(instruction))
	{
		// Insert incoming edges

		HierarchicalVisitor::VisitIn(instruction);

		// Insert outgoing edges for branching (label target)

		auto statementBlock = m_statementMap.at(instruction);
		auto [predicate, negate] = branchInstruction->GetPredicate();

		m_graph->InsertEdge(statementBlock, m_labelMap.at(branchInstruction->GetLabel()), predicate, negate);

		// Predicated branches have an edge to the next block

		if (branchInstruction->HasPredicate())
		{
			m_previousBlock = statementBlock;
		}
		else
		{
			m_previousBlock = nullptr;
		}
	}
	else if (instruction->HasPredicate())
	{
		// Remove predicate (now part of the block)

		instruction->SetPredicate(nullptr);

		// Other predicate instructions cause internal control flow

		m_previousBlock = nullptr;
	}
	else
	{
		// Insert incoming edges for all other instruction kinds

		HierarchicalVisitor::VisitIn(instruction);
	}

	return false;
}

bool ControlFlowBuilder::VisitIn(LabelStatement *statement)
{
	// Do nothing

	auto label = statement->GetLabel();
	auto statementBlock = m_labelMap.at(label);
	if (m_previousBlock != nullptr && m_previousBlock != statementBlock)
	{
		m_previousBlock->AddStatement(new BranchInstruction(label));
		m_graph->InsertEdge(m_previousBlock, statementBlock);
	}
	m_previousBlock = statementBlock;
	return false;
}

}
}
