#pragma once

#include <string>
#include <deque>

#include "Utils/Graph.h"

#include "SASS/Tree/Tree.h"

namespace SASS {
namespace Analysis {

using DependencyGraphNode = Instruction *;
class BlockDependencyGraph
{
public:
	BlockDependencyGraph(BasicBlock *block) : m_block(block)
	{
		auto size = block->GetInstructions().size();
		m_nodes.reserve(size);
	}

	enum DependencyKind {
		ReadWrite          = (1 << 0),
		WriteRead          = (1 << 1),
		WriteReadPredicate = (1 << 2),
		WriteWrite         = (1 << 3)
	};

	SASS_ENUM_FRIEND(DependencyKind)

	struct Node;
	struct Edge
	{
	public:
		Edge(Node& startNode, Node& endNode, DependencyKind dependencies)
			: m_startNode(startNode), m_endNode(endNode), m_dependencies(dependencies) {}

		const DependencyGraphNode& GetStartInstruction() const { return m_startNode.GetInstruction(); }
		const DependencyGraphNode& GetEndInstruction() const { return m_endNode.GetInstruction(); }

		const Node& GetStartNode() const { return m_startNode; }
		Node& GetStartNode() { return m_startNode; }

		const Node& GetEndNode() const { return m_endNode; }
		Node& GetEndNode() { return m_endNode; }

		DependencyKind GetDependencies() const { return m_dependencies; }
		DependencyKind& GetDependencies() { return m_dependencies; }

		void SetDependencies(DependencyKind dependencies) { m_dependencies = dependencies; }

	private:
		Node& m_startNode;
		Node& m_endNode;
		DependencyKind m_dependencies;
	};

	// For each instruction maintain scheduling information directly in the node:
	//   - Earliest virtual clock cycle for execution (updated when each parent is scheduled)
	//   - Stall count required if scheduled next
	// Together these give the legal execution time for the instruction.
	//
	// To account for variable latency dependencies, also maintain the same information
	// with the expected execution time and the active barrier state.
	//   - Read/write barriers
	//   - Barrier position (used for partial barriers)

	struct ScheduleProperties
	{
	public:
		std::uint32_t GetHeuristic() const { return m_heuristic; }
		void SetHeuristic(std::uint32_t heuristic) { m_heuristic = heuristic; }

		std::uint32_t GetAvailableTime() const { return m_time; }
		void SetAvailableTime(std::uint32_t time) { m_time = time; }

		std::uint32_t GetAvailableStall() const { return m_stall; }
		void SetAvailableStall(std::uint32_t stall) { m_stall = stall; }

		std::uint32_t GetBarrierTime() const { return m_barrierTime; }
		void SetBarrierTime(std::uint32_t barrierTime) { m_barrierTime = barrierTime; }

		std::uint32_t GetBarrierStall() const { return m_barrierStall; }
		void SetBarrierStall(std::uint32_t barrierStall) { m_barrierStall = barrierStall; }

		std::uint32_t GetDependencyCount() const { return m_dependencyCount; }
		void SetDependencyCount(std::uint32_t dependencyCount) { m_dependencyCount = dependencyCount; }

		std::uint16_t GetReadDependencyBarrier() const { return m_readDependencyBarrier; }
		void SetReadDependencyBarrier(std::uint16_t barrier) { m_readDependencyBarrier = barrier; }

		std::uint16_t GetWriteDependencyBarrier() const { return m_writeDependencyBarrier; }
		void SetWriteDependencyBarrier(std::uint16_t barrier) { m_writeDependencyBarrier = barrier; }

	private:
		std::uint32_t m_heuristic = 0;

		std::uint32_t m_time = 0;
		std::uint32_t m_stall = 1;
		std::uint32_t m_barrierTime = 0;
		std::uint32_t m_barrierStall = 1;

		std::uint32_t m_dependencyCount = 0;
		std::uint16_t m_writeDependencyBarrier = 0;
		std::uint16_t m_readDependencyBarrier = 0;
	};

	struct Node
	{
	public:
		Node(const DependencyGraphNode& instruction) : m_instruction(instruction)
		{
			m_incomingEdges.reserve(8);
			m_outgoingEdges.reserve(8);
		}

		const DependencyGraphNode& GetInstruction() const { return m_instruction; }

		const ScheduleProperties& GetScheduleProperties() const { return m_properties; }
		ScheduleProperties& GetScheduleProperties() { return m_properties; }

		std::size_t GetInDegree() const { return m_incomingEdges.size(); }
		std::size_t GetOutDegree() const { return m_outgoingEdges.size(); }

		const std::vector<std::reference_wrapper<Edge>>& GetIncomingEdges() const { return m_incomingEdges; }
		std::vector<std::reference_wrapper<Edge>>& GetIncomingEdges() { return m_incomingEdges; }

		const std::vector<std::reference_wrapper<Edge>>& GetOutgoingEdges() const { return m_outgoingEdges; }
		std::vector<std::reference_wrapper<Edge>>& GetOutgoingEdges() { return m_outgoingEdges; }

		void InsertIncomingEdge(Edge& edge) { m_incomingEdges.emplace_back(edge); }
		void InsertOutgoingEdge(Edge& edge) { m_outgoingEdges.emplace_back(edge); }

	private:
		DependencyGraphNode m_instruction;
		ScheduleProperties m_properties;

		std::vector<std::reference_wrapper<Edge>> m_incomingEdges;
		std::vector<std::reference_wrapper<Edge>> m_outgoingEdges;
	};

	// Nodes

	unsigned int GetNodeCount() const { return m_nodes.size(); }

	Node& InsertNode(const DependencyGraphNode& node)
	{
		return m_nodes.emplace(node, node).first->second;
	}

	// Edges

 	void InsertEdge(Node& sourceNode, Node& destinationNode, DependencyKind dependency)
	{
		// Augment existing edge with new dependency if one exists

		for (auto& edgeRef : sourceNode.GetOutgoingEdges())
		{
			auto& edge = edgeRef.get();
			if (edge.GetEndInstruction() == destinationNode.GetInstruction())
			{
				edge.GetDependencies() |= dependency;
				return;
			}
		}

		// Create a new edge - shared between the source/destination for dependency

		auto& edge = m_edges.emplace_back(sourceNode, destinationNode, dependency);

		sourceNode.InsertOutgoingEdge(edge);
		destinationNode.InsertIncomingEdge(edge);
	}
 	
	// Traversal

	template <typename F> 
	void ReverseTopologicalOrderBFS(F function)
	{
		// Construct the topological sorting structure
		//     Queue: store the current nodes 0 out-degree
		//     Edges: count the out-degree of each node

		std::deque<Node *> queue;
		robin_hood::unordered_flat_map<Node *, unsigned int> edges;

		edges.reserve(m_nodes.size());

		for (auto& [instruction, node] : m_nodes)
		{
			auto count = node.GetOutDegree();
			if (count == 0)
			{
				queue.push_back(&node);
			}
			else
			{
				edges.emplace(&node, count);
			}
		}

		// Perform the topological sort

		while (!queue.empty())
		{
			auto node = queue.front();
			queue.pop_front();

			// Apply the given function

			function(*node);

			// Process all predecessors of the node

			for (auto& edge : node->GetIncomingEdges())
			{
				auto& predecessor = edge.get().GetStartNode();
				if (--edges.at(&predecessor) == 0)
				{
					queue.push_back(&predecessor);
				}
			}
		}
	}

	// Formatting

	std::string ToDOTString() const;

private:
	BasicBlock *m_block = nullptr;

	robin_hood::unordered_node_map<DependencyGraphNode, Node> m_nodes;
	std::deque<Edge> m_edges;
};

SASS_ENUM_INLINE(BlockDependencyGraph, DependencyKind)

}
}
