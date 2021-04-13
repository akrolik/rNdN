#pragma once

#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Utils {

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
	unsigned int GetNodeCount() const { return m_nodes.size(); }
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

	enum Traversal {
		Preorder,
		Postorder
	};

	template<typename F>
	bool DFS(const T& start, F function, Traversal order = Traversal::Preorder) const
	{
		// Construct DFA ordering structure

		std::stack<T> outStack;
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

				if (order == Traversal::Preorder)
				{
					if (function(node))
					{
						return true;
					}
				}
				else if (order == Traversal::Postorder)
				{
					outStack.push(node);
				}

				// Maintain the visited structure and add successors

				visited.insert(node);
				for (auto successor : GetSuccessors(node))
				{
					stack.push(successor);
				}
			}
		}

		if (order == Traversal::Postorder)
		{
			while (!outStack.empty())
			{
				auto node = outStack.top();
				outStack.pop();

				if (function(node))
				{
					return true;
				}
			}
		}

		return false;
	}

	struct OrderContextDFS
	{
		OrderContextDFS(std::stack<T>& stack, std::unordered_map<T, unsigned int>& edges) : order(stack), edges(edges) {}

		std::stack<T>& order;
		std::unordered_map<T, unsigned int>& edges;
	};

	struct OrderContextBFS
	{
		OrderContextBFS(std::queue<T>& queue, std::unordered_map<T, unsigned int>& edges) : order(queue), edges(edges) {}

		std::queue<T>& order;
		std::unordered_map<T, unsigned int>& edges;
	};

	template <typename F> 
	void TopologicalOrderDFS(F function) const
	{
		// Construct the topological sorting structure
		//     Stack: store the current nodes 0 in-degree
		//     Edges: count the in-degree of each node

		std::stack<T> stack;
		std::unordered_map<T, unsigned int> edges;

		OrderContextDFS context(stack, edges);

		// Perform the topological sort

		while (!stack.empty())
		{
			auto& node = stack.top();
			stack.pop();

			// Apply the given function

			if (function(context, node))
			{
				// Process all successors of the node

				ProcessSuccessors(context, node);
			}
		}
	}

	template <typename F> 
	void TopologicalOrderBFS(F function) const
	{
		// Construct the topological sorting structure
		//     Queue: store the current nodes 0 in-degree
		//     Edges: count the in-degree of each node

		std::queue<T> queue;
		std::unordered_map<T, unsigned int> edges;

		OrderContextBFS context(queue, edges);

		InitializeOrderContext(context);

		// Perform the topological sort

		while (!queue.empty())
		{
			auto& node = queue.front();
			queue.pop();

			// Apply the given function

			if (function(context, node))
			{
				// Process all successors of the node

				ProcessSuccessors(context, node);
			}
		}
	}

	template <typename F> 
	void ReverseTopologicalOrderDFS(F function) const
	{
		// Construct the topological sorting structure
		//     Queue: store the current nodes 0 out-degree
		//     Edges: count the out-degree of each node

		std::stack<T> stack;
		std::unordered_map<T, unsigned int> edges;

		OrderContextDFS context(stack, edges);

		InitializeReverseOrderContext(context);

		// Perform the topological sort

		while (!stack.empty())
		{
			auto node = stack.top();
			stack.pop();

			// Apply the given function

			if (function(context, node))
			{
				// Process all predecessors of the node

				ProcessPredecessors(context, node);
			}
		}
	}
	template <typename F> 
	void ReverseTopologicalOrderBFS(F function) const
	{
		// Construct the topological sorting structure
		//     Queue: store the current nodes 0 out-degree
		//     Edges: count the out-degree of each node

		std::queue<T> queue;
		std::unordered_map<T, unsigned int> edges;

		OrderContextBFS context(queue, edges);

		InitializeReverseOrderContext(context);

		// Perform the topological sort

		while (!queue.empty())
		{
			auto node = queue.front();
			queue.pop();

			// Apply the given function

			if (function(context, node))
			{
				// Process all predecessors of the node

				ProcessPredecessors(context, node);
			}
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

	template<class O>
	void InitializeOrderContext(O& context) const
	{
		// Initialization with root nodes and count for incoming edges of each node

		for (auto& node : GetNodes())
		{
			auto count = GetLinearInDegree(node);
			if (count == 0)
			{
				context.order.push(node);
			}
			context.edges.insert({node, count});
		}
	}

	template<class O>
	void InitializeReverseOrderContext(O& context) const
	{
		// Initialization with root nodes and count for incoming edges of each node

		for (auto& node : GetNodes())
		{
			auto count = GetLinearOutDegree(node);
			if (count == 0)
			{
				context.order.push(node);
			}
			context.edges.insert({node, count});
		}
	}

	template<class O>
	void ProcessSuccessors(O& context, const T& node) const
	{
		auto& order = context.order;
		auto& edges = context.edges;

		// Decrease the degree of all successors, adding them to the queue
		// if they are at the front of the ordering

		for (const auto& successor : GetSuccessors(node))
		{
			edges.at(successor)--;
			if (edges.at(successor) == 0)
			{
				order.push(successor);
			}
		}
	}

	template<class O>
	void ProcessPredecessors(O& context, const T& node) const
	{
		auto& order = context.order;
		auto& edges = context.edges;

		// Decrease the degree of all predecessors, adding them to the queue
		// if they are at the front of the ordering

		for (const auto& predecessor : GetPredecessors(node))
		{
			edges.at(predecessor)--;
			if (edges.at(predecessor) == 0)
			{
				order.push(predecessor);
			}
		}
	}

	std::unordered_set<T> m_nodes;

	std::unordered_map<T, std::unordered_set<T>> m_successors;
	std::unordered_map<T, std::unordered_set<T>> m_predecessors;

	using EdgeType = std::pair<T, T>;
	struct EdgeHash
	{
		inline std::size_t operator()(const EdgeType& pair) const
		{
			return (std::hash<T>()(pair.first) * 31 + std::hash<T>()(pair.second));
		}
	};
};

}
