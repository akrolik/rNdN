#include "Analysis/Dependency/Overlay/DependencyOverlayPrinter.h"

#include "Analysis/Dependency/Overlay/DependencyOverlay.h"

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
		// Get a unique name for the statement and construct the string

		auto name = "s_" + std::to_string(++m_nameIndex);
		m_nameMap[node] = name;

		Indent();
		m_string << name <<"[label=\"" << HorseIR::PrettyPrinter::PrettyString(node, true) << "\", shape=rectangle";

		// Bold GPU capable statements

		auto graph = overlay->GetGraph();
		if (graph->IsGPUNode(node))
		{
			m_string << ", style=bold";
		}
		else if (graph->IsGPULibraryNode(node))
		{
			m_string << ", style=\"bold,diagonals,filled\"";
		}
			
		m_string << "];" << std::endl;
	}

	// Traverse all child overlays

	for (const auto& child : overlay->GetChildren())
	{
		Indent();
		m_string << "subgraph cluster_" << std::to_string(++m_nameIndex) << "{" << std::endl;
		m_indent++;

		Indent();

		// Fill overlays which are GPU kernels

		if (child->IsGPU() && !overlay->IsGPU())
		{
			m_string << "style=\"filled, dashed\";" << std::endl;
		}
		else
		{
			m_string << "style=dashed;" << std::endl;
		}
		Indent();
		m_string << "label=\"" << child->GetName() << "\";" << std::endl;

		child->Accept(*this);

		m_indent--;
		Indent();
		m_string << "}" << std::endl;
	}
}

void DependencyOverlayPrinter::Visit(const FunctionDependencyOverlay *overlay)
{
	// Construct the DOT surrounding structure with a unique name for the cluster

	auto function = overlay->GetNode();
	auto name = function->GetName() + "_" + std::to_string(++m_nameIndex);
	m_string << "digraph " << name << " {" << std::endl;

	// Style the subgraphs with dashed borders

	m_indent++;

	// Visit all statements and child overlays

	DependencyOverlayConstVisitor::Visit(overlay);

	// Add all edges in the graph

	auto graph = overlay->GetGraph();
	for (const auto& node : graph->GetNodes())
	{
		for (const auto& dependency : graph->GetSuccessors(node))
		{
			Indent();
			m_string << m_nameMap[node] << " -> " << m_nameMap[dependency];

			// Label compatible edges with an asterisk

			if (graph->IsSynchronizedEdge(node, dependency))
			{	
				m_string << " [style=bold, label=\"*\"]";
			}
			m_string << ";" << std::endl;
		}
	}

	// Tag the graph and end the structure

	Indent();
	m_indent--;

	m_string << "label=\"" << HorseIR::PrettyPrinter::PrettyString(function, true) << "\";" << std::endl;
	m_string << "}";
}

void DependencyOverlayPrinter::Visit(const IfDependencyOverlay *overlay)
{
	DependencyOverlayConstVisitor::Visit(overlay);
}

void DependencyOverlayPrinter::Visit(const WhileDependencyOverlay *overlay)
{
	DependencyOverlayConstVisitor::Visit(overlay);
}

void DependencyOverlayPrinter::Visit(const RepeatDependencyOverlay *overlay)
{
	DependencyOverlayConstVisitor::Visit(overlay);
}

}
