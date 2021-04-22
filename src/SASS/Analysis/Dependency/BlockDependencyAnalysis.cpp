#include "SASS/Analysis/Dependency/BlockDependencyAnalysis.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace SASS {
namespace Analysis {

void BlockDependencyAnalysis::Build(BasicBlock *block)
{
	// Construct dependency graph. In particular for SASS pay attention to:
	//   - System registers RZ/PT
	//   - Carry modifiers CC/X
	//   - Predicated instructions
	//   - Control dependencies (e.g. SSY/SYNC)

	auto timeAnalysis_start = Utils::Chrono::Start("Block dependency analysis '" + block->GetName() + "'");

	m_graph = new BlockDependencyGraph(block);
	m_currentSet.clear();
	m_controlInstruction = nullptr;

	m_readMap.clear();
	m_writeMap.clear();

	for (auto& instruction : block->GetInstructions())
	{
		m_graph->InsertNode(instruction);
		m_instruction = instruction;

		instruction->Accept(*this);
	}

	Utils::Chrono::End(timeAnalysis_start);

	// Debug printing

	if (Utils::Options::IsBackend_PrintAnalysis())
	{
		Utils::Logger::LogInfo("Block dependency graph: " + block->GetName());
		Utils::Logger::LogInfo(m_graph->ToDOTString(), 0, true, Utils::Logger::NoPrefix);
	}
}

void BlockDependencyAnalysis::Visit(Instruction *instruction)
{
	// Process source followed by destination operands

	m_destination = false;
	for (auto& operand : instruction->GetSourceOperands())
	{
		if (operand != nullptr)
		{
			operand->Accept(*this);
		}
	}

	m_destination = true;
	for (auto& operand : instruction->GetDestinationOperands())
	{
		if (operand != nullptr)
		{
			operand->Accept(*this);
		}
	}

	if (m_controlInstruction != nullptr)
	{
		if (m_graph->GetInDegree(instruction) == 0)
		{
			m_graph->InsertEdge(m_controlInstruction, instruction);
		}
	}

	m_currentSet.insert(instruction);
}

void BlockDependencyAnalysis::Visit(PredicatedInstruction *instruction)
{
	if (auto predicate = instruction->GetPredicate())
	{
		// Process predicate as a source register
		
		auto destination = m_destination;
		m_destination = false;

		predicate->Accept(*this);

		m_destination = destination;
		m_predicated = true;
	}

	Visitor::Visit(instruction);
	m_predicated = false;
}

void BlockDependencyAnalysis::Visit(BRAInstruction *instruction)
{
	Visitor::Visit(instruction);
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(BRKInstruction *instruction)
{
	Visitor::Visit(instruction);
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(CONTInstruction *instruction)
{
	Visitor::Visit(instruction);
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(EXITInstruction *instruction)
{
	Visitor::Visit(instruction);
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(PBKInstruction *instruction)
{
	Visitor::Visit(instruction);
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(PCNTInstruction *instruction)
{
	Visitor::Visit(instruction);
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(RETInstruction *instruction)
{
	Visitor::Visit(instruction);
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(SSYInstruction *instruction)
{
	Visitor::Visit(instruction);
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(SYNCInstruction *instruction)
{
	Visitor::Visit(instruction);
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(DEPBARInstruction *instruction)
{
	Visitor::Visit(instruction);
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(BARInstruction *instruction)
{
	Visitor::Visit(instruction);
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(MEMBARInstruction *instruction)
{
	Visitor::Visit(instruction);
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(Address *address)
{
	// Process address register as source

	auto destination = m_destination;
	m_destination = false;

	address->GetBase()->Accept(*this);

	m_destination = destination;

	// Process pointed-to data

	if (auto value = address->GetBase()->GetValue(); value != Register::ZeroIndex)
	{
		BuildDataDependencies(value + DataOffset_Address);
	}
}

void BlockDependencyAnalysis::Visit(Register *reg)
{
	// Skip system register

	auto value = reg->GetValue();
	if (value == Register::ZeroIndex)
	{
		return;
	}

	// Build dependencies for each register used

	auto range = reg->GetRange();
	for (auto i = value; i < value + range; ++i)
	{
		BuildDataDependencies(i + DataOffset_Register);
	}
}

void BlockDependencyAnalysis::Visit(Predicate *reg)
{
	// Skip system registers

	if (reg->GetValue() == Predicate::TrueIndex)
	{
		return;
	}

	BuildDataDependencies(reg->GetValue() + DataOffset_Predicate);
}

void BlockDependencyAnalysis::Visit(CarryFlag *carry)
{
	BuildDataDependencies(DataOffset_Carry);
}

void BlockDependencyAnalysis::BuildControlDependencies(Instruction *controlInstruction)
{
	// Add edge from all prior instructions that must be complete

	for (auto& instruction : m_currentSet)
	{
		if (instruction == controlInstruction)
		{
			continue;
		}
		if (m_graph->GetOutDegree(instruction) == 0)
		{
			m_graph->InsertEdge(instruction, controlInstruction);
		}
	}

	m_controlInstruction = controlInstruction;
	m_currentSet.clear();
	m_currentSet.insert(controlInstruction);

	for (auto& [operand, instructions] : m_readMap)
	{
		instructions.insert(controlInstruction);
	}

	for (auto& [operand, instructions] : m_writeMap)
	{
		instructions.insert(controlInstruction);
	}
}

void BlockDependencyAnalysis::BuildDataDependencies(std::uint32_t operand)
{
	auto& reads = m_readMap[operand];
	auto& writes = m_writeMap[operand];

	if (m_destination)
	{
		// Get Read-Write dependencies

		for (auto& readInstruction : reads)
		{
			if (readInstruction != m_instruction)
			{
				m_graph->InsertEdge(readInstruction, m_instruction);
			}
		}

		// Get Write-Write dependencies

		for (auto& writeInstruction : writes)
		{
			if (writeInstruction != m_instruction)
			{
				m_graph->InsertEdge(writeInstruction, m_instruction);
			}
		}

		// Remove old reads and writes (if non-predicated), add new write

		if (!m_predicated)
		{
			writes.clear();
			reads.clear();
		}
		writes.insert(m_instruction);
	}
	else
	{
		// Get Write-Read dependencies

		for (auto& writeInstruction : writes)
		{
			if (writeInstruction != m_instruction)
			{
				m_graph->InsertEdge(writeInstruction, m_instruction);
			}
		}

		// Add read for this instruction

		reads.insert(m_instruction);
	}
}

}
}
