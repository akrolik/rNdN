#pragma once

#include "Utils/Graph.h"

#include "PTX/Tree/BasicBlock.h"
#include "PTX/Tree/Operands/Label.h"
#include "PTX/Tree/Operands/Variables/Registers/Register.h"
#include "PTX/Tree/Type.h"

#include <stack>
#include <string>

#include "Utils/Logger.h"

#include "Libraries/robin_hood.h"

namespace PTX {

template<class T> class FunctionDefinition;

namespace Analysis {

using ControlFlowNode = BasicBlock *;
class ControlFlowGraph : public Utils::Graph<ControlFlowNode>
{
public:
	ControlFlowGraph(FunctionDefinition<VoidType> *function) : m_function(function) {}

	// Nodes

	void InsertNode(const ControlFlowNode& node)
	{
		Utils::Graph<ControlFlowNode>::InsertNode(node);

		m_labelMap[node->GetLabel()] = node;
	}

	ControlFlowNode GetNode(const Label *label) const { return m_labelMap.at(label); }

	void RemoveNode(const ControlFlowNode& node) override
	{
		m_labelMap.erase(node->GetLabel());

		Utils::Graph<ControlFlowNode>::RemoveNode(node);
	}

	// Edges

	void InsertEdge(const ControlFlowNode& source, const ControlFlowNode& destination, const Register<PredicateType> *predicate = nullptr, bool negate = false)
	{
		Utils::Graph<ControlFlowNode>::InsertEdge(source, destination);

		// Add predicate to edge

		auto edge = std::make_pair(source, destination);
		m_edgeData[edge] = { predicate, negate };
	}

	const std::pair<const Register<PredicateType> *, bool>& GetEdgeData(const ControlFlowNode& source, const ControlFlowNode& destination) const
	{
		return m_edgeData.at({source, destination});
	}
 
	void RemoveEdge(const ControlFlowNode& source, const ControlFlowNode& destination) override
	{
		m_edgeData.erase({source, destination});

		Utils::Graph<ControlFlowNode>::RemoveEdge(source, destination);
	}

	// Single entry node

	bool IsEntryNode(const ControlFlowNode& node) const { return m_entryNode == node; }
	const ControlFlowNode& GetEntryNode() const { return m_entryNode; }

	void SetEntryNode(const ControlFlowNode& entryNode) { m_entryNode = entryNode; }

	// Collection of exit nodes

	bool IsExitNode(const ControlFlowNode& node) const
	{
		return (m_exitNodes.find(node) != m_exitNodes.end());
	}
	const robin_hood::unordered_set<ControlFlowNode>& GetExitNodes() const { return m_exitNodes; }

	void SetExitNodes(const robin_hood::unordered_set<ControlFlowNode>& exitNodes) { m_exitNodes = exitNodes; }
	void AddExitNode(const ControlFlowNode& exitNode) { m_exitNodes.insert(exitNode); }

	// Ordering

	template <typename F> 
	void LinearOrdering(F function) const
	{
		// Construct DFA ordering structure

		std::stack<ControlFlowNode> stack;
		robin_hood::unordered_set<ControlFlowNode> visited;

		// Initialize with the entry node

		if (m_entryNode == nullptr)
		{
			Utils::Logger::LogError("CFG ordering requires entry node");
		}

		stack.push(m_entryNode);

		while (!stack.empty())
		{
			auto node = stack.top();
			stack.pop();

			if (visited.find(node) == visited.end())
			{
				// Apply the given function, exit if returned true

				function(node);

				// Maintain the visited structure and add successors

				visited.insert(node);

				// Add successors to the stack, firstly the fallthrough edge (no predicate) then the branching edge

				const auto& successors = GetSuccessors(node);
				std::vector<ControlFlowNode> successorsVec(std::begin(successors), std::end(successors));

				if (successorsVec.size() == 1)
				{
					const auto& successor = successorsVec.at(0);
					stack.push(successor);
				}
				else if (successorsVec.size() == 2)
				{
					const auto& successor0 = successorsVec.at(0);
					const auto& successor1 = successorsVec.at(1);

					if (const auto& [predicate0, negate0] = GetEdgeData(node, successor0); predicate0 == nullptr)
					{
						stack.push(successor1);
						stack.push(successor0);
					}
					else if (const auto& [predicate1, negate1] = GetEdgeData(node, successor1); predicate1 == nullptr)
					{
						stack.push(successor0);
						stack.push(successor1);
					}
					else
					{
						Utils::Logger::LogError("CFG branching requires condition");
					}
				}
				else if (successorsVec.size() > 2)
				{
					Utils::Logger::LogError("CFG nodes requires either 1 or 2 successors");
				}
			}
		}
	}

	// Formatting

	std::string ToDOTString() const;

private:
	FunctionDefinition<VoidType> *m_function = nullptr;
	robin_hood::unordered_map<const Label *, ControlFlowNode> m_labelMap;

	using EdgeType = typename Utils::Graph<ControlFlowNode>::EdgeType;
	using EdgeHash = typename Utils::Graph<ControlFlowNode>::EdgeHash;

	robin_hood::unordered_map<EdgeType, std::pair<const Register<PredicateType> *, bool>, EdgeHash> m_edgeData;

	ControlFlowNode m_entryNode = nullptr;
	robin_hood::unordered_set<ControlFlowNode> m_exitNodes;
};

}
}
