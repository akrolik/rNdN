#pragma once

#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace Analysis {

template<typename T>
class Graph
{
public:
	void InsertNode(const T& node)
	{
		// Create if needed

		m_successors[node];
		m_predecessors[node];

		m_nodes.insert(node);
	}

	void InsertEdge(const T& source, const T& destination)
	{
		m_successors[source].insert(destination);
		m_predecessors[destination].insert(source);
	}

	const std::unordered_set<T>& GetNodes() const { return m_nodes; }
	bool ContainsNode(const T& node) const { return (m_nodes.find(node) != m_nodes.end()); }

	const std::unordered_set<T>& GetPredecessors(const T& node) const { return m_predecessors.at(node); }
	const std::unordered_set<T>& GetSuccessors(const T& node) const { return m_successors.at(node); }

	unsigned int GetInDegree(const T& node) const
	{
		return m_predecessors.at(node).size();
	}

	unsigned int GetOutDegree(const T& node) const
	{
		return m_successors.at(node).size();
	}

	struct OrderingContext
	{
		OrderingContext(std::queue<T>& queue, std::unordered_map<T, unsigned int>& edges) : queue(queue), edges(edges) {}

		std::queue<T>& queue;
		std::unordered_map<T, unsigned int>& edges;
	};

	template <typename F> 
	void TopologicalOrdering(F function) const
	{
		// Construct the topological sorting structure
		//     Queue: store the current nodes 0 in-degree
		//     Edges: count the in-degree of each node

		std::queue<T> queue;
		std::unordered_map<T, unsigned int> edges;

		OrderingContext context(queue, edges);

		// Initialization with root nodes and count for incoming edges of each node

		for (const auto& node : GetNodes())
		{
			auto count = GetLinearOutDegree(node);
			if (count == 0)
			{
				queue.push(node);
			}
			edges.insert({node, count});
		}

		// Perform the topological sort

		while (!queue.empty())
		{
			const auto& node = queue.front();
			queue.pop();

			// Apply the given function

			function(node);

			// Process all predecessors of the node

			ProcessPredecessors(context, node);
		}
	}

protected:
	virtual unsigned int GetLinearOutDegree(const T& node) const
	{
		return GetOutDegree(node);
	}

	void ProcessPredecessors(OrderingContext& context, const T& node) const
	{
		auto& queue = context.queue;
		auto& edges = context.edges;

		// Decrease the degree of all predecessors, adding them to the queue
		// if they are at the front of the ordering

		for (const auto& predecessor : GetPredecessors(node))
		{
			edges.at(predecessor)--;
			if (edges.at(predecessor) == 0)
			{
				queue.push(predecessor);
			}
		}
	}

	std::unordered_set<T> m_nodes;

	std::unordered_map<T, std::unordered_set<T>> m_successors;
	std::unordered_map<T, std::unordered_set<T>> m_predecessors;
};

}
