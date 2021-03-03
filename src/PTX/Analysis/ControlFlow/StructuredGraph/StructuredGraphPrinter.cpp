#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraphPrinter.h"

namespace PTX {
namespace Analysis {

void StructuredGraphPrinter::Indent()
{
	m_string << std::string(m_indent * Utils::Logger::IndentSize, ' ');
}

std::string StructuredGraphPrinter::PrettyString(const StructureNode *node)
{
	StructuredGraphPrinter printer;
	printer.m_indent = 0;
	printer.m_string.str("");

	// Initialize structure

	//TODO: digraph + name source
	std::string name = "test";

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

		// Add edge connecting current structure to next

		Indent();
		m_string << m_nameMap[node] << " -> " << m_nameMap[next] << ";" << std::endl;
	}
}

void StructuredGraphPrinter::Visit(const BranchStructure *structure)
{
	//TODO: Better organize Branch subgraphs

	// Print main block

	Indent();
	auto name = structure->GetBlock()->GetLabel()->GetName();
	m_string << "s_" << name <<"[label=\"" << name << "\", shape=rectangle];" << std::endl;

	m_nameMap[structure] = "s_" + name;

	// Begin branch structure

	Indent();
	m_string << "subgraph cluster_" << name << " {" << std::endl;
	m_indent++;

	Indent();
	m_string << "style=dashed;" << std::endl;
	Indent();
	m_string << "label=\"Branch: " << name << "\";" << std::endl;

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
	}
	if (auto falseBranch = structure->GetFalseBranch())
	{
		Indent();
		m_string << m_nameMap[structure] << " -> " << m_nameMap[falseBranch] << ";" << std::endl;
	}
}

void StructuredGraphPrinter::Visit(const ExitStructure *structure)
{
	//TODO: Exit print
	ConstStructuredGraphVisitor::Visit(structure);
}

void StructuredGraphPrinter::Visit(const LoopStructure *structure)
{
	//TODO: Loop print
	ConstStructuredGraphVisitor::Visit(structure);
}

void StructuredGraphPrinter::Visit(const SequenceStructure *structure)
{
	// Add first node

	Indent();
	auto name = structure->GetBlock()->GetLabel()->GetName();
	m_string << "s_" << name <<"[label=\"" << name << "\", shape=rectangle];" << std::endl;

	m_nameMap[structure] = "s_" + name;

	// Process next structure

	ConstStructuredGraphVisitor::Visit(structure);
}

}
}
