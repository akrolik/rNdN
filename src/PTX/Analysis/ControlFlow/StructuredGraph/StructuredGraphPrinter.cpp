#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraphPrinter.h"

namespace PTX {
namespace Analysis {

void StructuredGraphPrinter::Indent()
{
	m_string << std::string(m_indent * Utils::Logger::IndentSize, ' ');
}

std::string StructuredGraphPrinter::PrettyString(const std::string& name, const StructureNode *node)
{
	StructuredGraphPrinter printer;
	printer.m_indent = 0;
	printer.m_string.str("");

	// Initialize structure

	printer.m_string << "digraph " << name << " {" << std::endl;
	printer.m_indent++;

	// Process nodes

	node->Accept(printer);

	// Close structure

	printer.Indent();
	printer.m_string << "label=\"" << name << "\";" << std::endl;
	printer.m_string << "}";

	return printer.m_string.str();
}

void StructuredGraphPrinter::Visit(const StructureNode *node)
{
	// Recurse following structures

	if (auto next = node->GetNext())
	{
		next->Accept(*this);
	}
}

void StructuredGraphPrinter::Visit(const BranchStructure *structure)
{
	auto clusterName = "branch" + std::to_string(m_index++);

	// Begin branch structure

	Indent();
	m_string << "subgraph cluster_" << clusterName << " {" << std::endl;
	m_indent++;

	Indent();
	m_string << "style=dashed;" << std::endl;
	Indent();
	m_string << "label=\"Branch\";" << std::endl;

	// Print main block

	Indent();
	auto name = structure->GetBlock()->GetLabel()->GetName();
	m_string << "s_" << name << "[label=\"" << name << "\", shape=rectangle];" << std::endl;

	m_nameMap[structure] = "s_" + name;

	// Print true/false branches

	if (auto trueBranch = structure->GetTrueBranch())
	{
		trueBranch->Accept(*this);
	}
	if (auto falseBranch = structure->GetFalseBranch())
	{
		falseBranch->Accept(*this);
	}

	// Close branch structure

	m_indent--;
	Indent();
	m_string << "}" << std::endl;

	// Process next structure

	ConstStructuredGraphVisitor::Visit(structure);

	// Add edges to branches
	
	if (auto trueBranch = structure->GetTrueBranch())
	{
		Indent();
		m_string << m_nameMap[structure] << " -> " << m_nameMap[trueBranch] << ";" << std::endl;

		if (auto next = structure->GetNext())
		{
			auto trueEnd = trueBranch;
			while (auto trueNext = trueEnd->GetNext())
			{
				trueEnd = trueNext;
			}

			Indent();
			m_string << m_nameMap[trueEnd] << " -> " << m_nameMap[next] << ";" << std::endl;
		}
	}
	if (auto falseBranch = structure->GetFalseBranch())
	{
		Indent();
		m_string << m_nameMap[structure] << " -> " << m_nameMap[falseBranch] << ";" << std::endl;

		if (auto next = structure->GetNext())
		{
			auto falseEnd = falseBranch;
			while (auto falseNext = falseEnd->GetNext())
			{
				falseEnd = falseNext;
			}

			Indent();
			m_string << m_nameMap[falseEnd] << " -> " << m_nameMap[next] << ";" << std::endl;
		}
	}
	if (structure->GetTrueBranch() == nullptr || structure->GetFalseBranch() == nullptr)
	{
		if (auto next = structure->GetNext())
		{
			Indent();
			m_string << m_nameMap[structure] << " -> " << m_nameMap[next] << ";" << std::endl;
		}
	}
}

void StructuredGraphPrinter::Visit(const ExitStructure *structure)
{
	Indent();
	auto name = structure->GetBlock()->GetLabel()->GetName();
	m_string << "s_" << name << "[label=\"E_" << name << "\", shape=rectangle];" << std::endl;

	m_nameMap[structure] = "s_" + name;

	// Process next structure

	ConstStructuredGraphVisitor::Visit(structure);
	
	// Add edge connecting block to the next

	if (auto next = structure->GetNext())
	{
		Indent();
		m_string << m_nameMap[structure] << " -> " << m_nameMap[next] << ";" << std::endl;
	}
}

void StructuredGraphPrinter::Visit(const LoopStructure *structure)
{
	// Begin loop structure

	auto clusterName = "loop" + std::to_string(m_index++);

	Indent();
	m_string << "subgraph cluster_" << clusterName << " {" << std::endl;
	m_indent++;

	Indent();
	m_string << "style=dashed;" << std::endl;
	Indent();
	m_string << "label=\"Loop\";" << std::endl;

	// Print body and copy naming (loop header is the structure name)

	if (auto body = structure->GetBody())
	{
		body->Accept(*this);

		m_nameMap[structure] = m_nameMap[body];
	}

	// Close loop structure

	m_indent--;
	Indent();
	m_string << "}" << std::endl;

	// Process next structure

	ConstStructuredGraphVisitor::Visit(structure);

	// Add edges to exits and loop latch
	
	if (auto next = structure->GetNext())
	{
		for (const auto& exit : structure->GetExits())
		{
			Indent();
			m_string << m_nameMap[exit] << " -> " << m_nameMap[next] << ";" << std::endl;
		}
	}

	if (auto latch = structure->GetLatch())
	{
		Indent();
		m_string << m_nameMap[latch] << " -> " << m_nameMap[structure] << ";" << std::endl;
	}
}

void StructuredGraphPrinter::Visit(const SequenceStructure *structure)
{
	// Add first node

	Indent();
	auto name = structure->GetBlock()->GetLabel()->GetName();
	m_string << "s_" << name << "[label=\"" << name << "\", shape=rectangle];" << std::endl;

	m_nameMap[structure] = "s_" + name;

	// Process next structure

	ConstStructuredGraphVisitor::Visit(structure);

	// Add edge connecting current structure to next

	if (auto next = structure->GetNext())
	{
		Indent();
		m_string << m_nameMap[structure] << " -> " << m_nameMap[next] << ";" << std::endl;
	}
}

}
}
