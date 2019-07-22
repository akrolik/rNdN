#include "Transformation/Outliner/OutlinePartitioner.h"

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

namespace Transformation {

Analysis::CompatibilityOverlay *OutlinePartitioner::Partition(const Analysis::CompatibilityOverlay *overlay)
{
	// We assume every input overlay will be associated with exactly 1 output overlay

	overlay->Accept(*this);
	return m_currentOverlays.at(0);
}

Analysis::CompatibilityOverlay *OutlinePartitioner::GetChildOverlay(const std::vector<Analysis::CompatibilityOverlay *>& childOverlays, const HorseIR::Statement *statement)
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

void OutlinePartitioner::ProcessEdges(PartitionContext& context, const Analysis::CompatibilityOverlay *overlay, const HorseIR::Statement *statement, const std::unordered_set<const HorseIR::Statement *>& destinations, bool flipped)
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

			if (graph->IsCompatibleEdge(statement, destination, flipped) && !graph->IsBackEdge(statement, destination, flipped))
			{
				queue.push(destination);
				visited.insert(destination);
			}
		}
		else
		{
			const auto childOverlay = GetChildOverlay(m_currentOverlays, destination);
			const auto statementOverlay = GetChildOverlay(m_currentOverlays, statement);

			if (childOverlay == nullptr || childOverlay == statementOverlay)
			{
				// Skip edges to parent or sibling overlays, they will be handled via recursion

				continue;
			}

			// If the destination is in a child overlay, consider the overlay for expansion

			if (visited.find(childOverlay) != visited.end())
			{
				continue;
			}

			// Skip synchronized child overlays as they cannot be merged

			if (childOverlay->IsSynchronized())
			{
				continue;
			}

			//TODO: We need to check compatibility with the geometry of the OVERLAY!

			// Check only a single ingoing edge as it's all we need to initiate a connected component.
			// Compatibility with a GPU is checked via reducibility and kernel flag

			if (graph->IsCompatibleEdge(statement, destination, flipped))
			{
				queue.push(childOverlay);
				visited.insert(childOverlay);
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
	std::unordered_set<NodeType> visited;
	std::queue<NodeType> queue;

	PartitionContext context(visited, queue);

	for (const auto statement : overlay->GetStatements())
	{
		nodes.insert(statement);
	}

	for (const auto child : m_currentOverlays)
	{
		nodes.insert(child);
	}

	// Build the connected components for the graph

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

		Analysis::KernelCompatibilityOverlay *kernelOverlay = nullptr;

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
						// Check if we already have a kernel started, if yes extend, if no start

						if (kernelOverlay == nullptr)
						{
							kernelOverlay = new Analysis::KernelCompatibilityOverlay(graph, newOverlay);
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
							kernelOverlay = new Analysis::KernelCompatibilityOverlay(graph, newOverlay);
						}

						// Transfer the contents of the kernel to the current kernel overlay

						for (auto child : childOverlay->GetChildren())
						{
							kernelOverlay->AddChild(child);
						}

						for (auto statement : childOverlay->GetStatements())
						{
							kernelOverlay->InsertStatement(statement);
						}

						// Process all crossing edges of the overlay

						ProcessOverlayEdges(context, overlay, childOverlay);

						//TODO: Delete old kernel overlay
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

		kernelOverlay = nullptr;
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
	}
}

void OutlinePartitioner::Visit(const Analysis::KernelCompatibilityOverlay *overlay)
{
	CompatibilityOverlayConstVisitor::Visit(overlay);
}

void OutlinePartitioner::Visit(const Analysis::FunctionCompatibilityOverlay *overlay)
{
	overlay->GetBody()->Accept(*this);

	auto size = m_currentOverlays.size();
	auto bodyOverlay = m_currentOverlays.at(size - 1);

	m_currentOverlays.pop_back();

	auto functionOverlay = new Analysis::FunctionCompatibilityOverlay(overlay->GetNode(), overlay->GetGraph());
	functionOverlay->SetChildren({bodyOverlay});

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

	auto node = overlay->GetNode();
	auto graph = overlay->GetGraph();

	if (trueOverlay->IsGPU() && !trueOverlay->IsSynchronized() && elseOverlay->IsGPU() && !elseOverlay->IsSynchronized())
	{
		//TODO: Check if both branches compatible

		bool isCompatible = true;
		if (isCompatible)
		{
			auto kernelOverlay = new Analysis::KernelCompatibilityOverlay(graph);
			auto ifOverlay = new Analysis::IfCompatibilityOverlay(node, graph, kernelOverlay);

			auto newTrueOverlay = new Analysis::CompatibilityOverlay(graph, ifOverlay);
			auto newElseOverlay = new Analysis::CompatibilityOverlay(graph, ifOverlay);

			// Copy across the kernel internals into the true/else branches

			newTrueOverlay->SetChildren(trueOverlay->GetChildren());
			newTrueOverlay->SetStatements(trueOverlay->GetStatements());

			newElseOverlay->SetChildren(elseOverlay->GetChildren());
			newElseOverlay->SetStatements(elseOverlay->GetStatements());

			ifOverlay->SetChildren({newTrueOverlay, newElseOverlay});
			ifOverlay->InsertStatement(node);

			m_currentOverlays.push_back(kernelOverlay);

			// Deallocate the old branch and kernel overlays

			delete trueOverlay;
			delete elseOverlay;
		}
	}
	else
	{
		auto ifOverlay = new Analysis::IfCompatibilityOverlay(node, graph);

		ifOverlay->SetChildren({trueOverlay, elseOverlay});
		ifOverlay->InsertStatement(node);

		m_currentOverlays.push_back(ifOverlay);
	}
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

	//TODO: Check for back edge compatibility
	if (bodyOverlay->IsGPU() && !bodyOverlay->IsSynchronized())
	{
		// Create a new kernel overlay and embed the loop

		auto kernelOverlay = new Analysis::KernelCompatibilityOverlay(graph);
		auto loopOverlay = new T(node, graph, kernelOverlay);
		auto newBodyOverlay = new Analysis::CompatibilityOverlay(graph, loopOverlay);

		// Copy across the kernel internals into the loop

		newBodyOverlay->SetChildren(bodyOverlay->GetChildren());
		newBodyOverlay->SetStatements(bodyOverlay->GetStatements());

		loopOverlay->SetChildren({newBodyOverlay});
		loopOverlay->InsertStatement(node);

		m_currentOverlays.push_back(kernelOverlay);

		// Deallocate the old kernel overlay

		delete bodyOverlay;
	}
	else
	{
		// Construct a simple loop with the body

		auto loopOverlay = new T(node, graph);

		loopOverlay->SetChildren({bodyOverlay});
		loopOverlay->InsertStatement(node);

		m_currentOverlays.push_back(loopOverlay);
	}
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
