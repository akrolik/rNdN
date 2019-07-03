#include "Analysis/Dependency/DependencyOverlayPrinter.h"

#include "Analysis/Dependency/DependencyOverlay.h"

namespace Analysis {

void DependencyOverlayPrinter::Indent()
{
	for (unsigned int i = 0; i < m_indent; ++i)
	{
		m_string << "\t";
	}
}

std::string DependencyOverlayPrinter::PrettyString(const DependencyOverlay *overlay)
{
	DependencyOverlayPrinter printer;
	printer.m_string.str("");
	overlay->Accept(printer);
	return printer.m_string.str();
}

void DependencyOverlayPrinter::Visit(const DependencyOverlay *overlay)
{
	// Add all statements in the overlay to the DOT cluster

	for (const auto& node : overlay->GetStatements())
	{
		// Get a unique name for the node and construct the string

		auto nodeNode = "s_" + std::to_string(++m_nameIndex);
		m_nameMap[node] = nodeNode;

		Indent();
		m_string << nodeNode <<"[label=\"" << HorseIR::PrettyPrinter::PrettyString(node, true) << "\", shape=rectangle];" << std::endl;
	}

	// Traverse all child overlays

	for (auto& child : overlay->GetChildren())
	{
		Indent();
		m_string << "subgraph cluster_" << std::to_string(++m_nameIndex) << "{" << std::endl;
		m_indent++;

		child->Accept(*this);

		m_indent--;
		Indent();
		m_string << "}" << std::endl;
	}
}

void DependencyOverlayPrinter::Visit(const CompoundDependencyOverlay<HorseIR::Function> *overlay)
{
	// Construct the DOT surrounding structure with a unique name for the cluster

	auto function = overlay->GetNode();
	auto name = function->GetName() + "_" + std::to_string(++m_nameIndex);
	m_string << "digraph " << name << " {" << std::endl;

	// Style the subgraphs with dashed borders

	m_indent++;
	Indent();
	m_string << "style=dashed;" << std::endl;

	// Visit all nodes and child overlays

	DependencyOverlayConstVisitor::Visit(overlay);

	// Add all edges in the graph

	auto graph = overlay->GetGraph();
	for (const auto& node : graph->GetNodes())
	{
		for (const auto& dependentNode : graph->GetOutgoingEdges(node))
		{
			Indent();
			m_string << m_nameMap[node] << " -> " << m_nameMap[dependentNode] << ";" << std::endl;
		}
	}

	// Tag the graph and end the structure

	Indent();
	m_indent--;

	m_string << "label=\"" << HorseIR::PrettyPrinter::PrettyString(function, true) << "\";" << std::endl;
	m_string << "}";
}

void DependencyOverlayPrinter::Visit(const CompoundDependencyOverlay<HorseIR::IfStatement> *overlay)
{
	DependencyOverlayConstVisitor::Visit(overlay);
}

void DependencyOverlayPrinter::Visit(const CompoundDependencyOverlay<HorseIR::WhileStatement> *overlay)
{
	DependencyOverlayConstVisitor::Visit(overlay);
}

void DependencyOverlayPrinter::Visit(const CompoundDependencyOverlay<HorseIR::RepeatStatement> *overlay)
{
	DependencyOverlayConstVisitor::Visit(overlay);
}

}
