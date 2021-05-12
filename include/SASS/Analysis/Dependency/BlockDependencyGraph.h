#pragma once

#include <string>

#include "Utils/Graph.h"

#include "SASS/Tree/Tree.h"

namespace SASS {
namespace Analysis {

using DependencyGraphNode = Instruction *;
class BlockDependencyGraph : public Utils::Graph<DependencyGraphNode>
{
public:
	BlockDependencyGraph(BasicBlock *block) : m_block(block) {}

	enum class DependencyKind {
		ReadWrite,
		WriteRead,
		WriteWrite
	};

	// Nodes

	void InsertNode(const DependencyGraphNode& node, std::uint32_t value = 0)
	{
		Utils::Graph<DependencyGraphNode>::InsertNode(node);

		m_values[node] = value;
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

 	void InsertEdge(const DependencyGraphNode& source, const DependencyGraphNode& destination, DependencyKind dependency)
	{
		Utils::Graph<DependencyGraphNode>::InsertEdge(source, destination);

		// Add dependency to edge, extending the set if it already exists

		auto edge = std::make_pair(source, destination);
		m_edgeData[edge].insert(dependency);
	}

	const robin_hood::unordered_set<DependencyKind>& GetEdgeDependencies(const DependencyGraphNode& source, const DependencyGraphNode& destination) const
	{
		return m_edgeData.at({source, destination});
	}

	// Formatting

	std::string ToDOTString() const;

private:
	BasicBlock *m_block = nullptr;

	using EdgeType = typename Utils::Graph<DependencyGraphNode>::EdgeType;
	using EdgeHash = typename Utils::Graph<DependencyGraphNode>::EdgeHash;

	robin_hood::unordered_map<EdgeType, robin_hood::unordered_set<DependencyKind>, EdgeHash> m_edgeData;
	robin_hood::unordered_map<DependencyGraphNode, std::uint32_t> m_values;
};

}
}
