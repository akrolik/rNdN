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

	struct Edge
	{
	public:
		Edge(DependencyGraphNode start, DependencyGraphNode end, DependencyKind dependencies)
			: m_start(start), m_end(end), m_dependencies(dependencies) {}

		const DependencyGraphNode& GetStart() const { return m_start; }
		const DependencyGraphNode& GetEnd() const { return m_end; }

		DependencyKind GetDependencies() const { return m_dependencies; }
		DependencyKind& GetDependencies() { return m_dependencies; }

		void SetDependencies(DependencyKind dependencies) { m_dependencies = dependencies; }

	private:
		DependencyGraphNode m_start = nullptr;
		DependencyGraphNode m_end = nullptr;
		DependencyKind m_dependencies;
	};

	struct Node
	{
	public:
		Node(std::uint32_t value) : m_value(value)
		{
			m_incomingEdges.reserve(8);
			m_outgoingEdges.reserve(8);
		}

		std::size_t GetInDegree() const { return m_incomingEdges.size(); }
		std::size_t GetOutDegree() const { return m_outgoingEdges.size(); }

		std::uint32_t GetValue() const { return m_value; }
		void SetValue(std::uint32_t value) { m_value = value; }

		const std::vector<Edge *>& GetIncomingEdges() const { return m_incomingEdges; }
		const std::vector<Edge *>& GetOutgoingEdges() const { return m_outgoingEdges; }

		void InsertIncomingEdge(Edge *edge) { m_incomingEdges.push_back(edge); }
		void InsertOutgoingEdge(Edge *edge) { m_outgoingEdges.push_back(edge); }

	private:
		std::uint32_t m_value = 0;
		std::vector<Edge *> m_incomingEdges;
		std::vector<Edge *> m_outgoingEdges;
	};

	// Nodes

	const robin_hood::unordered_flat_map<DependencyGraphNode, Node>& GetNodes() const { return m_nodes; }
	unsigned int GetNodeCount() const { return m_nodes.size(); }
	bool ContainsNode(const DependencyGraphNode& node) const { return (m_nodes.find(node) != m_nodes.end()); }

	void InsertNode(const DependencyGraphNode& node, std::uint32_t value = 0)
	{
		m_nodes.emplace(node, value);
	}

	void SetNodeValue(const DependencyGraphNode& node, std::uint32_t value)
	{
		m_nodes.at(node).SetValue(value);
	}

	std::uint32_t GetNodeValue(const DependencyGraphNode& node) const
	{
		return m_nodes.at(node).GetValue();
	}

	// Edges

 	void InsertEdge(const DependencyGraphNode& source, const DependencyGraphNode& destination, DependencyKind dependency)
	{
		// Augment existing edge with new dependency if one exists

		auto& node = m_nodes.at(source);
		for (auto& edge : node.GetOutgoingEdges())
		{
			if (edge->GetEnd() == destination)
			{
				edge->GetDependencies() |= dependency;
				return;
			}
		}

		// Create a new edge - shared between the source/destination for value

		auto edge = new Edge(source, destination, dependency);

		node.InsertOutgoingEdge(edge);
		m_nodes.at(destination).InsertIncomingEdge(edge);
	}

	std::size_t GetInDegree(const DependencyGraphNode& node) const
	{
		return m_nodes.at(node).GetInDegree();
	}

	std::size_t GetOutDegree(const DependencyGraphNode& node) const
	{
		return m_nodes.at(node).GetOutDegree();
	}

	const std::vector<Edge *>& GetIncomingEdges(const DependencyGraphNode& destination) const
	{
		return m_nodes.at(destination).GetIncomingEdges();
	}

	const std::vector<Edge *>& GetOutgoingEdges(const DependencyGraphNode& source) const
	{
		return m_nodes.at(source).GetOutgoingEdges();
	}

	// Traversal

	template <typename F> 
	void ReverseTopologicalOrderBFS(F function)
	{
		// Construct the topological sorting structure
		//     Queue: store the current nodes 0 out-degree
		//     Edges: count the out-degree of each node

		std::deque<DependencyGraphNode> queue;
		robin_hood::unordered_map<DependencyGraphNode, unsigned int> edges;

		edges.reserve(m_nodes.size());

		for (auto& [instruction, node] : m_nodes)
		{
			auto count = node.GetOutDegree();
			if (count == 0)
			{
				queue.push_back(instruction);
			}
			else
			{
				edges.emplace(instruction, count);
			}
		}

		// Perform the topological sort

		while (!queue.empty())
		{
			auto instruction = queue.front();
			queue.pop_front();

			// Apply the given function

			auto& node = m_nodes.at(instruction);
			function(instruction, node);

			// Process all predecessors of the node

			for (const auto& edge : node.GetIncomingEdges())
			{
				auto predecessor = edge->GetStart();
				if (--edges.at(predecessor) == 0)
				{
					queue.push_back(predecessor);
				}
			}
		}
	}

	// Formatting

	std::string ToDOTString() const;

private:
	BasicBlock *m_block = nullptr;

	robin_hood::unordered_flat_map<DependencyGraphNode, Node> m_nodes;
};

SASS_ENUM_INLINE(BlockDependencyGraph, DependencyKind)

}
}
