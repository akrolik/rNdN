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

	m_block = block;

	for (auto& instruction : block->GetInstructions())
	{
		if (m_graph == nullptr)
		{
			InitializeSection();
		}

		m_instruction = instruction;
		instruction->Accept(*this);
	}

	Utils::Chrono::End(timeAnalysis_start);

	// Debug printing

	if (Utils::Options::IsBackend_PrintAnalysis())
	{
		Utils::Logger::LogInfo("Block dependency graphs: " + block->GetName());
		for (auto graph : m_graphs)
		{
			Utils::Logger::LogInfo(graph->ToDOTString(), 0, true, Utils::Logger::NoPrefix);
		}
	}
}

void BlockDependencyAnalysis::InitializeSection()
{
	m_graph = new BlockDependencyGraph(m_block);
	m_graphs.push_back(m_graph);

	m_readMap.clear();
	m_writeMap.clear();
}

void BlockDependencyAnalysis::Visit(Instruction *instruction)
{
	// Process source followed by destination operands

	if (!m_predicated)
	{
		m_graph->InsertNode(instruction);
	}

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
}

void BlockDependencyAnalysis::Visit(PredicatedInstruction *instruction)
{
	m_graph->InsertNode(instruction);

	if (auto predicate = instruction->GetPredicate())
	{
		// Process predicate as a source register
		
		auto destination = m_destination;
		m_destination = false;

		Visit(predicate);

		m_destination = destination;
		m_predicated = true;
	}

	Visit(static_cast<Instruction *>(instruction));

	m_predicated = false;
}

void BlockDependencyAnalysis::Visit(BRAInstruction *instruction)
{
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(BRKInstruction *instruction)
{
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(CONTInstruction *instruction)
{
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(EXITInstruction *instruction)
{
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(PBKInstruction *instruction)
{
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(PCNTInstruction *instruction)
{
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(RETInstruction *instruction)
{
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(SSYInstruction *instruction)
{
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(SYNCInstruction *instruction)
{
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(DEPBARInstruction *instruction)
{
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(BARInstruction *instruction)
{
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(MEMBARInstruction *instruction)
{
	BuildControlDependencies(instruction);
}

void BlockDependencyAnalysis::Visit(Address *address)
{
	// Process address register as source

	auto destination = m_destination;
	m_destination = false;

	Visit(address->GetBase());

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
	// Create a new section just for this instruction

	if (m_graph->GetNodeCount() > 0)
	{
		InitializeSection();
	}

	m_graph->InsertNode(controlInstruction);
	m_graph = nullptr;
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
				m_graph->InsertEdge(readInstruction, m_instruction, BlockDependencyGraph::DependencyKind::ReadWrite);
			}
		}

		// Get Write-Write dependencies

		for (auto& writeInstruction : writes)
		{
			if (writeInstruction != m_instruction)
			{
				m_graph->InsertEdge(writeInstruction, m_instruction, BlockDependencyGraph::DependencyKind::WriteWrite);
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
				m_graph->InsertEdge(writeInstruction, m_instruction, BlockDependencyGraph::DependencyKind::WriteRead);
			}
		}

		// Add read for this instruction

		reads.insert(m_instruction);
	}
}

}
}
