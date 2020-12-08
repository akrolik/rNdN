#pragma once

#include "Utils/Graph.h"

#include "PTX/Tree/Tree.h"
#include "PTX/Analysis/ControlFlow/BasicBlock.h"

#include <string>

namespace PTX {
namespace Analysis {

using ControlFlowNode = const BasicBlock *;
class ControlFlowGraph : public Utils::Graph<ControlFlowNode>
{
public:
	ControlFlowGraph(const FunctionDefinition<VoidType> *function) : m_function(function) {}

	// Nodes

	void InsertNode(const ControlFlowNode& node, const Label *label)
	{
		Utils::Graph<ControlFlowNode>::InsertNode(node);

		m_labelMap[label] = node;
	}

	ControlFlowNode GetNode(const Label *label) const { return m_labelMap.at(label); }

	void RemoveNode(const ControlFlowNode& node) override
	{
		//TODO: Erase BB map
		// m_labelMap.erase(node);

		Utils::Graph<ControlFlowNode>::RemoveNode(node);
	}

	// Edges

	void InsertEdge(const ControlFlowNode& source, const ControlFlowNode& destination, const Register<PredicateType> *predicate = nullptr)
	{
		Utils::Graph<ControlFlowNode>::InsertEdge(source, destination);

		// Add predicate to edge

		auto edge = std::make_pair(source, destination);
		m_edgeData[edge] = predicate;
	}

	const Register<PredicateType> *GetEdgeData(const ControlFlowNode& source, const ControlFlowNode& destination) const
	{
		return m_edgeData.at({source, destination});
	}
 
	void RemoveEdge(const ControlFlowNode& source, const ControlFlowNode& destination) override
	{
		m_edgeData.erase({source, destination});

		Utils::Graph<ControlFlowNode>::RemoveEdge(source, destination);
	}

	// Formatting

	std::string ToDOTString() const;

private:
	const FunctionDefinition<VoidType> *m_function = nullptr;
	std::unordered_map<const Label *, ControlFlowNode> m_labelMap;

	using EdgeType = typename Utils::Graph<ControlFlowNode>::EdgeType;
	using EdgeHash = typename Utils::Graph<ControlFlowNode>::EdgeHash;

	std::unordered_map<EdgeType, const Register<PredicateType> *, EdgeHash> m_edgeData;
};

}
}
