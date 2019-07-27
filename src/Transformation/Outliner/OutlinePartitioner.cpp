#include "Transformation/Outliner/OutlinePartitioner.h"

#include "Analysis/Compatibility/CompatibilityAnalysis.h"
#include "Analysis/Compatibility/Geometry/GeometryUtils.h"

namespace Transformation {

Analysis::CompatibilityOverlay *OutlinePartitioner::Partition(const Analysis::CompatibilityOverlay *overlay)
{
	// We assume every input overlay will be associated with exactly 1 output overlay

	overlay->Accept(*this);
	return m_currentOverlays.at(0);
}

Analysis::CompatibilityOverlay *OutlinePartitioner::GetChildOverlay(const std::vector<Analysis::CompatibilityOverlay *>& childOverlays, const HorseIR::Statement *statement) const
{
	// Return the top-level child overlay which contains the statement

	for (const auto child : childOverlays)
	{
		// Check the current overlay for the statement

		if (child->ContainsStatement(statement))
		{
			return child;
		}
	}

	// Check all children overlays, note that we want the highest *containing* overlay, not the actual overlay

	for (const auto child : childOverlays)
	{
		if (GetChildOverlay(child->GetChildren(), statement) != nullptr)
		{
			return child;
		}
	}

	// Else, this is contained in a sibling or parent overlay

	return nullptr;
}

void OutlinePartitioner::ProcessEdges(PartitionContext& context, const Analysis::CompatibilityOverlay *overlay, const HorseIR::Statement *statement, const std::unordered_set<const HorseIR::Statement *>& destinations, bool incoming)
{
	// Check all connected in/out edges for compatibility, skipping back edges

	auto graph = overlay->GetGraph();

	auto& visited = context.visited;
	auto& queue = context.queue;

	for (const auto destination : destinations)
	{
		if (overlay->ContainsStatement(destination))
		{
			// If the destination is in the current overlay, consider the edge for expansion

			if (visited.find(destination) != visited.end())
			{
				continue;
			}

			// Check that the edge is compatible and not a back edge, they will by the structure overlay

			if (graph->IsCompatibleEdge(statement, destination, incoming) && !graph->IsBackEdge(statement, destination, incoming))
			{
				queue.push(destination);
				visited.insert(destination);
			}
		}
		else
		{
			const auto statementOverlay = GetChildOverlay(m_currentOverlays, statement);
			const auto destinationOverlay = GetChildOverlay(m_currentOverlays, destination);

			if (destinationOverlay == nullptr || destinationOverlay == statementOverlay)
			{
				// Skip edges to parent or sibling overlays, they will be handled via recursion

				continue;
			}

			// If the destination is in a child overlay, consider the overlay for expansion

			if (visited.find(destinationOverlay) != visited.end())
			{
				continue;
			}

			// Check if the overlay is GPU compatible, which is a requirement for compatibility

			if (!destinationOverlay->IsGPU())
			{
				continue;
			}

			// Skip synchronized destination overlays if the edge is incoming (outgoing from the overlay)

			if (incoming && destinationOverlay->IsSynchronized())
			{
				continue;
			}

			// Check the statement geometry compatibility with the geometry of the overlay

			auto statementGeometry = m_geometryAnalysis.GetGeometry(statement);
			auto overlayGeometry = m_overlayGeometries.at(destinationOverlay);

			if (Analysis::CompatibilityAnalysis::IsCompatible(statementGeometry, overlayGeometry))
			{
				queue.push(destinationOverlay);
				visited.insert(destinationOverlay);
			}
		}
	}
}

void OutlinePartitioner::ProcessOverlayEdges(PartitionContext& context, const Analysis::CompatibilityOverlay *container, const Analysis::CompatibilityOverlay *overlay)
{
	auto graph = container->GetGraph();

	// Process all edges from the current overlay

	for (const auto statement : overlay->GetStatements())
	{
		ProcessEdges(context, container, statement, graph->GetOutgoingEdges(statement), false);
		ProcessEdges(context, container, statement, graph->GetIncomingEdges(statement), true);
	}

	// Process all edges from the child overlays

	for (const auto child : overlay->GetChildren())
	{
		ProcessOverlayEdges(context, container, child);
	}
}

void OutlinePartitioner::Visit(const Analysis::CompatibilityOverlay *overlay)
{
	// Recursively traverse all child overlays, we will build the partition bottom-up

	const auto currentOverlays = m_currentOverlays;
	m_currentOverlays.clear();

	for (const auto& child : overlay->GetChildren())
	{
		child->Accept(*this);
	}

	// Construct the set of nodes for the overlay, including statements and children overlays

	std::unordered_set<NodeType> nodes;

	for (const auto statement : overlay->GetStatements())
	{
		nodes.insert(statement);
	}

	for (const auto child : m_currentOverlays)
	{
		nodes.insert(child);
	}

	// Build the connected components for the graph

	std::unordered_set<NodeType> visited;
	std::queue<NodeType> queue;
	PartitionContext context(visited, queue);

	const auto graph = overlay->GetGraph();
	auto newOverlay = new Analysis::CompatibilityOverlay(graph);

	for (const auto& node : nodes)
	{
		// Check if the node is already visited

		if (visited.find(node) != visited.end())
		{
			continue;
		}

		// Add to the queue for processing and connected nodes

		queue.push(node);
		visited.insert(node);

		// Construct a connected component from the queue elements

		Analysis::CompatibilityOverlay *kernelOverlay = nullptr;

		while (!queue.empty())
		{
			const auto& node = queue.front();
			queue.pop();

			std::visit(overloaded {

				// If the node is a statement, check if it is GPU compatible - in which case we construct a kernel
				// and continue building. If not, add to the main overlay (function or body of control structure)

				[&](const HorseIR::Statement *statement)
				{
					if (graph->IsGPUNode(statement))
					{
						// Check if we already have a kernel started. If yes extend, if no start

						if (kernelOverlay == nullptr)
						{
							kernelOverlay = new Analysis::CompatibilityOverlay(graph, newOverlay);
							kernelOverlay->SetGPU(true);
						}

						// Add the statement to the kernel

						kernelOverlay->InsertStatement(statement);

						// Process all edges of the statement

						ProcessEdges(context, overlay, statement, graph->GetOutgoingEdges(statement), false);
						ProcessEdges(context, overlay, statement, graph->GetIncomingEdges(statement), true);
					}
					else
					{
						// CPU statements get added to the main overlay

						newOverlay->InsertStatement(statement);
					}        
				},
				[&](Analysis::CompatibilityOverlay *childOverlay)
				{
					if (childOverlay->IsGPU())
					{
						// Check if we already have a kernel started, if yes extend, if no start

						if (kernelOverlay == nullptr)
						{
							kernelOverlay = new Analysis::CompatibilityOverlay(graph, newOverlay);
							kernelOverlay->SetGPU(true);
						}

						// Transfer the contents of the kernel to the current kernel overlay

						kernelOverlay->AddChild(childOverlay);

						// Process all crossing edges of the overlay

						ProcessOverlayEdges(context, overlay, childOverlay);
					}
					else
					{
						// CPU overlays get added to the main overlay

						newOverlay->AddChild(childOverlay);
					}
				}},
				node
			);
		}

		if (kernelOverlay != nullptr)
		{
			// Compute the effective geometry of the kernel

			const Analysis::Geometry *geometry = nullptr;
			for (const auto statement : kernelOverlay->GetStatements())
			{
				geometry = Analysis::GeometryUtils::MaxGeometry(geometry, m_geometryAnalysis.GetGeometry(statement));
			}

			for (const auto child : kernelOverlay->GetChildren())
			{
				geometry = Analysis::GeometryUtils::MaxGeometry(geometry, m_overlayGeometries.at(child));
			}
			m_overlayGeometries[kernelOverlay] = geometry;

			// Reset the kernel for the next component, by construction the parent/child relationship is already set

			kernelOverlay = nullptr;
		}
	}

	m_currentOverlays = currentOverlays;

	//TODO: Merge newOverlay kernels

	if (newOverlay->IsReducible())
	{
		m_currentOverlays.push_back(newOverlay->GetChild(0));
		delete newOverlay;
	}
	else
	{
		m_currentOverlays.push_back(newOverlay);
		m_overlayGeometries[newOverlay] = new Analysis::CPUGeometry();
	}
}

void OutlinePartitioner::Visit(const Analysis::FunctionCompatibilityOverlay *overlay)
{
	overlay->GetBody()->Accept(*this);

	auto size = m_currentOverlays.size();
	auto bodyOverlay = m_currentOverlays.at(size - 1);

	m_currentOverlays.pop_back();

	auto functionOverlay = new Analysis::FunctionCompatibilityOverlay(overlay->GetNode(), overlay->GetGraph());
	functionOverlay->SetChildren({bodyOverlay});

	m_overlayGeometries[functionOverlay] = m_overlayGeometries[bodyOverlay];

	m_currentOverlays.push_back(functionOverlay);
}

void OutlinePartitioner::Visit(const Analysis::IfCompatibilityOverlay *overlay)
{
	overlay->GetTrueBranch()->Accept(*this);
	overlay->GetElseBranch()->Accept(*this);

	auto size = m_currentOverlays.size();
	auto trueOverlay = m_currentOverlays.at(size - 2);
	auto elseOverlay = m_currentOverlays.at(size - 1);

	m_currentOverlays.pop_back();
	m_currentOverlays.pop_back();

	// Determine if the if statement is GPU compatible. Check that both branches are GPU overlays.
	// We don't check synchronization as there is no looping and both may exit and synchronize independently

	bool kernel = false;

	if (trueOverlay->IsGPU() && elseOverlay->IsGPU())
	{
		// Check geometries of both branches are compatible

		auto trueGeometry = m_overlayGeometries.at(trueOverlay);
		auto elseGeometry = m_overlayGeometries.at(elseOverlay);

		kernel = Analysis::CompatibilityAnalysis::IsCompatible(trueGeometry, elseGeometry);
	}

	// Create the if statement overlay with the statement, bodies, and GPU flag, setting the resulting geometry

	auto node = overlay->GetNode();
	auto graph = overlay->GetGraph();

	auto ifOverlay = new Analysis::IfCompatibilityOverlay(node, graph);
	ifOverlay->SetGPU(kernel);

	ifOverlay->SetChildren({trueOverlay, elseOverlay});
	ifOverlay->InsertStatement(node);

	if (kernel)
	{
		// If both geometries are compatible, find the max operating range for the entire if statement

		auto trueGeometry = m_overlayGeometries.at(trueOverlay);
		auto elseGeometry = m_overlayGeometries.at(elseOverlay);

		m_overlayGeometries[ifOverlay] = Analysis::GeometryUtils::MaxGeometry(trueGeometry, elseGeometry);
	}
	else
	{
		m_overlayGeometries[ifOverlay] = new Analysis::CPUGeometry();
	}

	m_currentOverlays.push_back(ifOverlay);
}

template<typename T>
void OutlinePartitioner::VisitLoop(const T *overlay)
{
	// Process the loop body

	overlay->GetBody()->Accept(*this);

	auto size = m_currentOverlays.size();
	auto bodyOverlay = m_currentOverlays.at(size - 1);

	m_currentOverlays.pop_back();

	auto node = overlay->GetNode();
	auto graph = overlay->GetGraph();

	// Check that the body overlay is GPU capable and has no synchronized statements
	
	bool kernel = false;

	if (bodyOverlay->IsGPU() && !bodyOverlay->IsSynchronized())
	{
		//TODO: Check for back edge compatibility
		kernel = true;
	}

	// Construct a loop with the body and GPU flag, propagating the geometry

	auto loopOverlay = new T(node, graph);
	loopOverlay->SetGPU(kernel);

	loopOverlay->SetChildren({bodyOverlay});
	loopOverlay->InsertStatement(node);

	m_overlayGeometries[loopOverlay] = m_overlayGeometries[bodyOverlay];

	m_currentOverlays.push_back(loopOverlay);
}

void OutlinePartitioner::Visit(const Analysis::WhileCompatibilityOverlay *overlay)
{
	VisitLoop(overlay);
}

void OutlinePartitioner::Visit(const Analysis::RepeatCompatibilityOverlay *overlay)
{
	VisitLoop(overlay);
}

}
