#include "Transformation/Outliner/OutlinePartitioner.h"

namespace Transformation {

Analysis::CompatibilityOverlay *OutlinePartitioner::Partition(const Analysis::CompatibilityOverlay *overlay)
{
	overlay->Accept(*this);
	return m_overlay;
}

unsigned int OutlinePartitioner::GetSortingOutDegree(const Analysis::CompatibilityOverlay *overlay, const HorseIR::Statement *statement)
{
	auto graph = overlay->GetGraph();

	unsigned int count = 0;
	for (auto& destination : graph->GetOutgoingEdges(statement))
	{
		// Ignore back edges, we will handle them separately

		if (graph->IsBackEdge(statement, destination))
		{
			continue;
		}

		count++;
	}
	return count;
}

void OutlinePartitioner::Visit(const Analysis::CompatibilityOverlay *overlay)
{
	// Recursively traverse all child overlays, we will build the partition bottom-up

	for (const auto child : overlay->GetChildren())
	{
		child->Accept(*this);
	}

	// Initialize the stack with the root statement and overlay nodes

 	std::stack<const PartitionElement *> entryStack;

	auto graph = overlay->GetGraph();
	for (const auto& statement : overlay->GetStatements())
	{
		// Get the number of edges excluding crossing and back edges

		auto element = new StatementElement(statement);
		auto edges = GetSortingOutDegree(overlay, statement);
		if (edges == 0)
		{
			entryStack.push(element);
		}
		m_edges.insert({element, edges});
	}

	while (!entryStack.empty())
	{
		// Traverse the partition elements topologically and construct a new overlay greedily

		Analysis::KernelCompatibilityOverlay *kernelOverlay = nullptr;

		std::stack<const PartitionElement *> overlayStack;
		overlayStack.push(entryStack.top());
		entryStack.pop();

		while (!overlayStack.empty())
		{
			auto element = overlayStack.top();
			overlayStack.pop();

			switch (element->GetKind())
			{
				case PartitionElement::Kind::Statement:
				{
					// Extract the statement from the element

					auto statement = static_cast<const StatementElement *>(element)->GetStatement();
					if (graph->IsGPUNode(statement))
					{
						// Check if we already have a kernel started, if yes extend, if no start

						bool inserted = false;
						if (kernelOverlay != nullptr)
						{
							// Check compatibility with all dependent statements in the new overlay

							bool linked = false;
							bool compatible = true;
							for (const auto& destination : graph->GetOutgoingEdges(statement))
							{
								// Check the edge requires a compatibility check for the kernel. By construction this should also skip back edges

								if (!kernelOverlay->ContainsStatement(destination))
								{
									continue;
								}
								
								// Check if this is linked into the kernel, stops horizontal slicing

								linked = true;

								// Check compatibility, requires all compatible to return true

								compatible &= graph->IsCompatibleEdge(statement, destination);
							}

							// If the statement is compatible with the kernel add, otherwise stack for the next kernel start

							if (compatible && linked)
							{
								kernelOverlay->InsertStatement(statement);
								inserted = true;
							}
							else
							{
								entryStack.push(element);
							}
						}
						else
						{
							// Start a new kernel with the given statement

							kernelOverlay = new Analysis::KernelCompatibilityOverlay(m_overlay->GetGraph(), m_overlay);
							kernelOverlay->InsertStatement(statement);
							inserted = true;
						}

						// Decrease the cached in-degree of all dependency sources

						if (inserted)
						{
							for (auto& destination : graph->GetIncomingEdges(statement))
							{
								auto element = new StatementElement(destination);
								auto count = --m_edges.at(element);
								if (count == 0)
								{
									overlayStack.push(element);
								}
							}
						}
					}
					else
					{
						// If the statement is non GPU and we have an open kernel context, stack and handle later.
						// If no kernel context is open, add to the function

						if (kernelOverlay == nullptr)
						{
							m_overlay->InsertStatement(statement);

							for (auto& destination : graph->GetIncomingEdges(statement))
							{
								auto element = new StatementElement(destination);
								auto count = --m_edges.at(element);
								if (count == 0)
								{
									overlayStack.push(element);
								}
							}
						}
						else
						{
							entryStack.push(element);
						}
					}
					break;
				}
			}
		}   
	}
}

void OutlinePartitioner::Visit(const Analysis::FunctionCompatibilityOverlay *overlay)
{
	//TODO: We should have a better way of constructing the overlays
	m_overlay = new Analysis::FunctionCompatibilityOverlay(overlay->GetNode(), overlay->GetGraph());

	Analysis::CompatibilityOverlayConstVisitor::Visit(overlay);
}

}
