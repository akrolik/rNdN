#include "Analysis/Compatibility/CompatibilityAnalysis.h"

#include "Analysis/Geometry/GeometryUtils.h"
#include "Analysis/Shape/ShapeUtils.h"
#include "Analysis/Dependency/Overlay/DependencyOverlay.h"
#include "Analysis/Dependency/Overlay/DependencyOverlayPrinter.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Analysis {

void CompatibilityAnalysis::Analyze(const DependencyOverlay *overlay)
{
	auto timeCompatibility_start = Utils::Chrono::Start();
	overlay->Accept(*this);
	auto timeCompatibility = Utils::Chrono::End(timeCompatibility_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_outline_graph))
	{
		Utils::Logger::LogInfo("Compatibility graph");

		auto compatibilityString = Analysis::DependencyOverlayPrinter::PrettyString(m_currentOverlays.at(0));
		Utils::Logger::LogInfo(compatibilityString, 0, true, Utils::Logger::NoPrefix);
	}
	Utils::Logger::LogTiming("Compatibility analysis", timeCompatibility);
}

DependencyOverlay *CompatibilityAnalysis::GetKernelOverlay(const DependencySubgraph *subgraph, const DependencySubgraphNode& node, DependencyOverlay *parentOverlay) const
{
	// Get or create the kernel for the node

	auto kernelOverlay = GetSuccessorsKernelOverlay(subgraph, node);

	// If the statement cannot be added to the successors overlay (if a unique overlay exists), create a new kernel

	if (kernelOverlay == nullptr)
	{
		kernelOverlay = new DependencyOverlay(parentOverlay->GetGraph(), parentOverlay);
		kernelOverlay->SetGPU(true);
	}

	return kernelOverlay;
}

DependencyOverlay *CompatibilityAnalysis::GetSuccessorsKernelOverlay(const DependencySubgraph *subgraph, const DependencySubgraphNode& node) const
{
	// Check all successors for compatibility, and that they reside in the same kernel

	DependencyOverlay *kernelOverlay = nullptr;

	for (const auto& successor : subgraph->GetSuccessors(node))
	{
		// Ignore back edges, they are handled by the loop overlay

		if (subgraph->IsBackEdge(node, successor))
		{
			continue;
		}

		// Check the sucessor for incoming or outgoing synchronization which prevents merge

		if (subgraph->IsSynchronizedEdge(node, successor))
		{
			return nullptr;
		}

		// Check all successors reside in the same kernel

		auto search = m_kernelMap.find(successor);
		if (search == m_kernelMap.end())
		{
			return nullptr;
		}
		
		auto successorOverlay = search->second;
		if (kernelOverlay == nullptr)
		{
			kernelOverlay = successorOverlay;
		}
		else if (kernelOverlay != successorOverlay)
		{
			return nullptr;
		}

		// Check the compatibility between the current node and the successor

		auto nodeGeometry = GetGeometry(node);
		auto successorGeometry = GetGeometry(successor);

		if (!IsCompatible(nodeGeometry, successorGeometry))
		{
			return nullptr;
		}
	}

	return kernelOverlay;
}

const Shape *CompatibilityAnalysis::GetGeometry(const DependencySubgraphNode& node) const
{
	const Shape *geometry = nullptr;
	std::visit(overloaded
	{
		[&](const HorseIR::Statement *statement)
		{
			geometry = m_geometryAnalysis.GetGeometry(statement);
		},
		[&](const DependencyOverlay *overlay)
		{
			// We receive the old graph node, but fetch the geometry of the new overlay

			geometry = m_overlayGeometries.at(m_overlayMap.at(overlay));
		}},
		node
	);
	return geometry;
}

const Shape *CompatibilityAnalysis::BuildGeometry(const DependencyOverlay *overlay) const
{
	// Build the geometry of the overlay

	if (overlay->IsGPU())
	{
		// Compute the effective geometry of the kernel for all statements and children overlays

		const Shape *geometry = nullptr;
		for (const auto& statement : overlay->GetStatements())
		{
			geometry = GeometryUtils::MaxGeometry(geometry, m_geometryAnalysis.GetGeometry(statement));
		}
		for (const auto& childOverlay : overlay->GetChildren())
		{
			geometry = GeometryUtils::MaxGeometry(geometry, m_overlayGeometries.at(childOverlay));
		}
		return geometry;
	}
	else
	{
		return new WildcardShape();
	}
}

bool CompatibilityAnalysis::IsCompatible(const Shape *source, const Shape *destination) const
{
	if (*source == *destination)
	{
		return true;
	}

	if (source->GetKind() != destination->GetKind())
	{
		return false;
	}

	switch (source->GetKind())
	{
		case Shape::Kind::Vector:
		{
			auto sourceSize = ShapeUtils::GetShape<VectorShape>(source)->GetSize();
			auto destinationSize = ShapeUtils::GetShape<VectorShape>(destination)->GetSize();
			return IsCompatible(sourceSize, destinationSize);
		}
		case Shape::Kind::List:
		{
			auto sourceList = ShapeUtils::GetShape<ListShape>(source);
			auto destinationList = ShapeUtils::GetShape<ListShape>(destination);

			if (sourceList->GetListSize() != destinationList->GetListSize())
			{
				return false;
			}

			auto sourceCell = ShapeUtils::MergeShapes(sourceList->GetElementShapes());
			auto destinationCell = ShapeUtils::MergeShapes(destinationList->GetElementShapes());

			return IsCompatible(sourceCell, destinationCell);
		}
	}

	return false;
}

bool CompatibilityAnalysis::IsCompatible(const Shape::Size *source, const Shape::Size *destination) const
{
	// Check for initialization compatibility

	if (ShapeUtils::IsSize<Shape::InitSize>(source))
	{
		return true;
	}

	// Allow scalars to be merged anywhere

	if (ShapeUtils::IsScalarSize(source) || ShapeUtils::IsScalarSize(destination))
	{
		return true;
	}

	// Check for compression compatibility

	if (!ShapeUtils::IsSize<Shape::CompressedSize>(destination))
	{
		return false;
	}

	auto unmaskedSize = ShapeUtils::GetSize<Shape::CompressedSize>(destination)->GetSize();
	return (*source == *unmaskedSize);
}

void CompatibilityAnalysis::Visit(const DependencyOverlay *overlay)
{
	// The child overlay construction is bottom-up, so we temporarily store sibling overlays to store later

	const auto currentOverlays = m_currentOverlays;
	m_currentOverlays.clear();

	// Perform the topological sort and construct the new overlay

	auto graph = overlay->GetGraph();
	auto newOverlay = new DependencyOverlay(graph);

	const auto subgraph = overlay->GetSubgraph();
	subgraph->TopologicalOrdering([&](const DependencySubgraphNode& node)
	{
		std::visit(overloaded
		{
			[&](const HorseIR::Statement *statement)
			{
				// If the node is a statement, check if it is GPU compatible - in which case
				// find the appropriate kernel (new if needed). If not, add it to the CPU overlay

				if (graph->IsGPUNode(statement))
				{
					// Find or construct the overlay for the statement and add the statement

					auto kernelOverlay = GetKernelOverlay(subgraph, node, newOverlay);
					kernelOverlay->InsertStatement(statement);
					m_kernelMap[node] = kernelOverlay;
				}
				else
				{
					// CPU statements get added to the main overlay

					newOverlay->InsertStatement(statement);
				}        
			},
			[&](const DependencyOverlay *childOverlay)
			{
				// Process the child overlay and reconstruct bottom-up

				childOverlay->Accept(*this);

				auto processedChildOverlay = m_currentOverlays.back();
				if (processedChildOverlay->IsGPU())
				{
					// Add the child kernel to the GPU capable successor kernel. If it does not exist,
					// the child kernel can be added directly to the container as it is self contained

					auto kernelOverlay = GetKernelOverlay(subgraph, node, newOverlay);
					kernelOverlay->AddChild(processedChildOverlay);
					m_kernelMap[node] = kernelOverlay;
				}
				else
				{
					// CPU overlays get added to the main overlay

					newOverlay->AddChild(processedChildOverlay);
				}
			}},
			node
		);
	});

	// Update the geometry calculations of all child overlays

	for (auto& childOverlay : newOverlay->GetChildren())
	{
		if (childOverlay->IsGPU())
		{
			m_overlayGeometries[childOverlay] = BuildGeometry(childOverlay);
		}
	}

	// Finalize the new overlay and compute the geometry if needed

	m_currentOverlays = currentOverlays;

	if (newOverlay->IsReducible())
	{
		m_currentOverlays.push_back(newOverlay->GetChild(0));
		delete newOverlay;
	}
	else
	{
		m_overlayGeometries[newOverlay] = BuildGeometry(newOverlay);
		m_currentOverlays.push_back(newOverlay);
	}
}

void CompatibilityAnalysis::Visit(const FunctionDependencyOverlay *overlay)
{
	// Process the function body and grab the resulting overlay

	overlay->GetBody()->Accept(*this);

	auto size = m_currentOverlays.size();
	auto bodyOverlay = m_currentOverlays.at(size - 1);

	m_currentOverlays.pop_back();

	// Construct a function overlay and propagate the resulting geometry

	auto functionOverlay = new FunctionDependencyOverlay(overlay->GetNode(), overlay->GetGraph());
	functionOverlay->SetChildren({bodyOverlay});
	m_currentOverlays.push_back(functionOverlay);

	m_overlayGeometries[functionOverlay] = m_overlayGeometries.at(bodyOverlay);
}

void CompatibilityAnalysis::Visit(const IfDependencyOverlay *overlay)
{
	if (overlay->HasElseBranch())
	{
		// Process the true/else branches and grab the resulting overlays

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

			kernel = IsCompatible(trueGeometry, elseGeometry);
		}

		// Create the if statement overlay with the statement, bodies, and GPU flag, setting the resulting geometry

		auto node = overlay->GetNode();
		auto graph = overlay->GetGraph();

		auto ifOverlay = new IfDependencyOverlay(node, graph);
		ifOverlay->SetGPU(kernel);
		ifOverlay->SetChildren({trueOverlay, elseOverlay});
		ifOverlay->InsertStatement(node);

		// Compute the geometry of the if statement overlay

		if (kernel)
		{
			// If both geometries are compatible, find the max operating range for the entire if statement

			auto trueGeometry = m_overlayGeometries.at(trueOverlay);
			auto elseGeometry = m_overlayGeometries.at(elseOverlay);

			m_overlayGeometries[ifOverlay] = GeometryUtils::MaxGeometry(trueGeometry, elseGeometry);
		}
		else
		{
			m_overlayGeometries[ifOverlay] = new WildcardShape();
		}

		m_overlayMap[overlay] = ifOverlay;
		m_currentOverlays.push_back(ifOverlay);
	}
	else
	{
		// Process the true branch and grab the resulting overlay

		overlay->GetTrueBranch()->Accept(*this);

		auto size = m_currentOverlays.size();
		auto trueOverlay = m_currentOverlays.at(size - 1);

		m_currentOverlays.pop_back();

		// Determine if the if statement is GPU compatible, check if the true branch is compatible
		// We don't check synchronization as there is no looping

		bool kernel = trueOverlay->IsGPU();

		// Create the if statement overlay with the statement, true body, and GPU flag, setting the resulting geometry

		auto node = overlay->GetNode();
		auto graph = overlay->GetGraph();

		auto ifOverlay = new IfDependencyOverlay(node, graph);
		ifOverlay->SetGPU(kernel);
		ifOverlay->SetChildren({trueOverlay});
		ifOverlay->InsertStatement(node);

		// Compute the geometry of the if statement overlay

		if (kernel)
		{
			m_overlayGeometries[ifOverlay] = m_overlayGeometries.at(trueOverlay);
		}
		else
		{
			m_overlayGeometries[ifOverlay] = new WildcardShape();
		}

		m_overlayMap[overlay] = ifOverlay;
		m_currentOverlays.push_back(ifOverlay);
	}
}

bool CompatibilityAnalysis::IsIterable(const DependencyOverlay *overlay) const
{
	// For each back edge, make sure it is iterable (not synchronized, and geometry compatible)

	auto subgraph = overlay->GetSubgraph();
	for (const auto& node : subgraph->GetNodes())
	{
		for (const auto& successor : subgraph->GetSuccessors(node))
		{
			if (subgraph->IsBackEdge(node, successor))
			{
				// Check synchronization

				if (subgraph->IsSynchronizedEdge(node, successor))
				{
					return false;
				}

				// Check geometry compatibility

				auto outGeometry = GetGeometry(node);
				auto inGeometry = GetGeometry(successor);

				if (!IsCompatible(outGeometry, inGeometry))
				{
					return false;
				}
			}
		}
	}
	return true;
}

template<typename T>
void CompatibilityAnalysis::VisitLoop(const T *overlay)
{
	// Process the loop body and grab the resulting overlay

	overlay->GetBody()->Accept(*this);

	auto size = m_currentOverlays.size();
	auto bodyOverlay = m_currentOverlays.at(size - 1);

	m_currentOverlays.pop_back();

	auto node = overlay->GetNode();
	auto graph = overlay->GetGraph();

	// Check that the body overlay is GPU capable and has no synchronized statements
	
	bool kernel = false;
	if (bodyOverlay->IsGPU())
	{
		kernel = IsIterable(overlay->GetBody());
	}

	// Construct a loop with the body and GPU flag

	auto loopOverlay = new T(node, graph);
	loopOverlay->SetGPU(kernel);
	loopOverlay->SetChildren({bodyOverlay});
	loopOverlay->InsertStatement(node);

	// Compute the geometry of the loop overlay, either propagating that of the body
	// or placing the loop on the CPU

	if (kernel)
	{
		m_overlayGeometries[loopOverlay] = m_overlayGeometries.at(bodyOverlay);
	}
	else
	{
		m_overlayGeometries[loopOverlay] = new WildcardShape();
	}

	m_overlayMap[overlay] = loopOverlay;
	m_currentOverlays.push_back(loopOverlay);
}

void CompatibilityAnalysis::Visit(const WhileDependencyOverlay *overlay)
{
	VisitLoop(overlay);
}

void CompatibilityAnalysis::Visit(const RepeatDependencyOverlay *overlay)
{
	VisitLoop(overlay);
}

}
