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
	BlockDependencyGraph(BasicBlock *block) : m_block(block) {}

	enum class DependencyKind {
		ReadWrite,
		WriteRead,
		WriteReadPredicate,
		WriteWrite
	};

	// Nodes

	const robin_hood::unordered_set<DependencyGraphNode>& GetNodes() const { return m_nodes; }
	unsigned int GetNodeCount() const { return m_nodes.size(); }
	bool ContainsNode(const DependencyGraphNode& node) const { return (m_nodes.find(node) != m_nodes.end()); }

	void InsertNode(const DependencyGraphNode& node, std::uint32_t value = 0)
	{
		m_nodes.insert(node);
		m_values.emplace(node, value);

		m_outgoingEdges[node];
		m_incomingEdges[node];
	}

	void SetNodeValue(const DependencyGraphNode& node, std::uint32_t value)
	{
		m_values.at(node) = value;
	}

	std::uint32_t GetNodeValue(const DependencyGraphNode& node) const
	{
		return m_values.at(node);
	}

	// Edges

	struct Edge
	{
	public:
		Edge(DependencyGraphNode start, DependencyGraphNode end, DependencyKind kind)
			: m_start(start), m_end(end), m_dependencies({kind}) {}

		DependencyGraphNode GetStart() const { return m_start; }
		DependencyGraphNode GetEnd() const { return m_end; }

		const std::vector<DependencyKind>& GetDependencies() const { return m_dependencies; }
		std::vector<DependencyKind>& GetDependencies() { return m_dependencies; }

		void SetDependencies(const std::vector<DependencyKind>& dependencies) { m_dependencies = dependencies; }

	private:
		DependencyGraphNode m_start = nullptr;
		DependencyGraphNode m_end = nullptr;
		std::vector<DependencyKind> m_dependencies;
	};

 	void InsertEdge(const DependencyGraphNode& source, const DependencyGraphNode& destination, DependencyKind dependency)
	{
		// Augment existing edge with new dependency if one exists

		for (auto& edge : m_outgoingEdges.at(source))
		{
			if (edge->GetEnd() == destination)
			{
				edge->GetDependencies().push_back(dependency);
				return;
			}
		}

		// Create a new edge - shared between the source/destination for value

		auto edge = new Edge(source, destination, dependency);

		m_outgoingEdges.at(source).push_back(edge);
		m_incomingEdges.at(destination).push_back(edge);
	}

	std::size_t GetInDegree(const DependencyGraphNode& node) const
	{
		return m_incomingEdges.at(node).size();
	}
	std::size_t GetOutDegree(const DependencyGraphNode& node) const
	{
		return m_outgoingEdges.at(node).size();
	}

	const std::vector<Edge *>& GetOutgoingEdges(const DependencyGraphNode& source) const
	{
		return m_outgoingEdges.at(source);
	}

	const std::vector<Edge *>& GetIncomingEdges(const DependencyGraphNode& destination) const
	{
		return m_incomingEdges.at(destination);
	}

	// Traversal

	template <typename F> 
	void ReverseTopologicalOrderBFS(F function) const
	{
		// Construct the topological sorting structure
		//     Queue: store the current nodes 0 out-degree
		//     Edges: count the out-degree of each node

		std::deque<DependencyGraphNode> queue;
		robin_hood::unordered_map<DependencyGraphNode, unsigned int> edges;

		for (auto& node : GetNodes())
		{
			auto count = GetOutDegree(node);
			if (count == 0)
			{
				queue.push_back(node);
			}
			edges.emplace(node, count);
		}

		// Perform the topological sort

		while (!queue.empty())
		{
			auto node = queue.front();
			queue.pop_front();

			// Apply the given function

			if (function(node))
			{
				// Process all predecessors of the node

				for (const auto& edge : m_incomingEdges.at(node))
				{
					auto predecessor = edge->GetStart();
					if (--edges.at(predecessor) == 0)
					{
						queue.push_back(predecessor);
					}
				}
			}
		}
	}

	// Formatting

	std::string ToDOTString() const;

private:
	BasicBlock *m_block = nullptr;

	robin_hood::unordered_set<DependencyGraphNode> m_nodes;
	robin_hood::unordered_map<DependencyGraphNode, std::uint32_t> m_values;

	robin_hood::unordered_map<DependencyGraphNode, std::vector<Edge *>> m_incomingEdges;
	robin_hood::unordered_map<DependencyGraphNode, std::vector<Edge *>> m_outgoingEdges;
};

}
}
