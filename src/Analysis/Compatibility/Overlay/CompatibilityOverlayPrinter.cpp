#include "Analysis/Compatibility/Overlay/CompatibilityOverlayPrinter.h"

#include "Analysis/Compatibility/Overlay/CompatibilityOverlay.h"

namespace Analysis {

void CompatibilityOverlayPrinter::Indent()
{
	for (unsigned int i = 0; i < m_indent; ++i)
	{
		m_string << "\t";
	}
}

std::string CompatibilityOverlayPrinter::PrettyString(const CompatibilityOverlay *overlay)
{
	CompatibilityOverlayPrinter printer;
	printer.m_string.str("");
	overlay->Accept(printer);
	return printer.m_string.str();
}

void CompatibilityOverlayPrinter::Visit(const CompatibilityOverlay *overlay)
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
			// Add diagonal marks on the corner if the node is synchronized

			if (graph->IsSynchronizedNode(node))
			{
				m_string << ", style=\"bold, diagonals\"";
			}
			else
			{
				m_string << ", style=bold";
			}
		}
			
		m_string << "];" << std::endl;
	}

	// Traverse all child overlays

	for (auto& child : overlay->GetChildren())
	{
		Indent();
		m_string << "subgraph cluster_" << std::to_string(++m_nameIndex) << "{" << std::endl;
		m_indent++;

		Indent();

		// Fill overlays which are GPU kernels

		if (child->IsGPU() && !overlay->IsGPU())
		{
			m_string << "style=filled;" << std::endl;
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

void CompatibilityOverlayPrinter::Visit(const FunctionCompatibilityOverlay *overlay)
{
	// Construct the DOT surrounding structure with a unique name for the cluster

	auto function = overlay->GetNode();
	auto name = function->GetName() + "_" + std::to_string(++m_nameIndex);
	m_string << "digraph " << name << " {" << std::endl;

	// Style the subgraphs with dashed borders

	m_indent++;

	// Visit all statements and child overlays

	CompatibilityOverlayConstVisitor::Visit(overlay);

	// Add all edges in the graph

	auto graph = overlay->GetGraph();
	for (const auto& node : graph->GetNodes())
	{
		for (const auto& dependency : graph->GetOutgoingEdges(node))
		{
			Indent();
			m_string << m_nameMap[node] << " -> " << m_nameMap[dependency];

			// Label compatible edges with an asterisk

			if (graph->IsCompatibleEdge(node, dependency))
			{	
				m_string << " [label=\"*\"]";
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

void CompatibilityOverlayPrinter::Visit(const IfCompatibilityOverlay *overlay)
{
	CompatibilityOverlayConstVisitor::Visit(overlay);
}

void CompatibilityOverlayPrinter::Visit(const WhileCompatibilityOverlay *overlay)
{
	CompatibilityOverlayConstVisitor::Visit(overlay);
}

void CompatibilityOverlayPrinter::Visit(const RepeatCompatibilityOverlay *overlay)
{
	CompatibilityOverlayConstVisitor::Visit(overlay);
}

}
