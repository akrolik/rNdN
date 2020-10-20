#pragma once

#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>

namespace HorseIR {
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

	virtual void RemoveNode(const T& node)
	{
		m_nodes.erase(node);

		const auto successors = m_successors.at(node);
		for (auto successor : successors)
		{
			RemoveEdge(node, successor);
		}

		const auto predecessors = m_predecessors.at(node);
		for (auto predecessor : predecessors)
		{
			RemoveEdge(predecessor, node);
		}
	}

	virtual void RemoveEdge(const T& source, const T& destination)
	{
		m_successors.at(source).erase(destination);
		m_predecessors.at(destination).erase(source);
	}

	void InsertEdge(const T& source, const T& destination)
	{
		m_successors[source].insert(destination);
		m_predecessors[destination].insert(source);
	}

	bool ContainsEdge(const T& source, const T& destination) const
	{
		const auto& successors = m_successors.at(source);
		return (successors.find(destination) != successors.end());
	}

	bool ContainsPath(const T& source, const T& destination) const
	{
		auto found = DFS(source, [&](const T& node)
		{
			return (destination == node);
		});

		return found;
	}

	template<typename F>
	bool DFS(const T& start, F function) const
	{
		// Construct DFA ordering structure

		std::stack<T> stack;
		std::unordered_set<T> visited;

		// Initialize with the given node

		stack.push(start);

		while (!stack.empty())
		{
			auto node = stack.top();
			stack.pop();

			if (visited.find(node) == visited.end())
			{
				// Apply the given function, exit if returned true

				if (function(node))
				{
					return true;
				}

				// Maintain the visited structure and add successors

				visited.insert(node);
				for (auto successor : GetSuccessors(node))
				{
					stack.push(successor);
				}
			}
		}
		return false;
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

		for (auto& node : GetNodes())
		{
			auto count = GetLinearInDegree(node);
			if (count == 0)
			{
				queue.push(node);
			}
			edges.insert({node, count});
		}

		// Perform the topological sort

		while (!queue.empty())
		{
			auto& node = queue.front();
			queue.pop();

			// Apply the given function

			auto status = function(context, node);

			// Process all successors of the node

			if (status)
			{
				ProcessSuccessors(context, node);
			}
		}
	}

	template <typename F> 
	void ReverseTopologicalOrdering(F function) const
	{
		// Construct the topological sorting structure
		//     Queue: store the current nodes 0 out-degree
		//     Edges: count the out-degree of each node

		std::queue<T> queue;
		std::unordered_map<T, unsigned int> edges;

		OrderingContext context(queue, edges);

		// Initialization with root nodes and count for incoming edges of each node

		for (auto& node : GetNodes())
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
			auto node = queue.front();
			queue.pop();

			// Apply the given function

			function(node);

			// Process all predecessors of the node

			ProcessPredecessors(context, node);
		}
	}

protected:
	virtual unsigned int GetLinearInDegree(const T& node) const
	{
		return GetInDegree(node);
	}

	virtual unsigned int GetLinearOutDegree(const T& node) const
	{
		return GetOutDegree(node);
	}

	void ProcessSuccessors(OrderingContext& context, const T& node) const
	{
		auto& queue = context.queue;
		auto& edges = context.edges;

		// Decrease the degree of all successors, adding them to the queue
		// if they are at the front of the ordering

		for (const auto& successor : GetSuccessors(node))
		{
			edges.at(successor)--;
			if (edges.at(successor) == 0)
			{
				queue.push(successor);
			}
		}
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
}
