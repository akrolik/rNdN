#pragma once

#include <unordered_set>
#include <utility>
#include <variant>

#include "Analysis/Utils/Graph.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

class DependencyOverlay;

using DependencySubgraphNode = std::variant<const HorseIR::Statement *, const DependencyOverlay *>;

class DependencySubgraph : public Graph<DependencySubgraphNode>
{
public:
	using Graph<DependencySubgraphNode>::Graph;

	void InsertEdge(const DependencySubgraphNode& source, const DependencySubgraphNode& destination, const std::unordered_set<const HorseIR::SymbolTable::Symbol *>& symbols, bool isBackEdge)
	{
		Graph<DependencySubgraphNode>::InsertEdge(source, destination);

		// Add symbol to edge, extending the set if it already exists

		auto edge = std::make_pair(source, destination);
		if (m_edgeData.find(edge) == m_edgeData.end())
		{
			m_edgeData[edge] = symbols;
		}
		else
		{
			m_edgeData.at(edge).insert(std::begin(symbols), std::end(symbols));
		}

		// Add to set of back edges if needed

		if (isBackEdge)
		{
			m_backEdges.insert({source, destination});
		}
	}

	bool IsBackEdge(const DependencySubgraphNode& source, const DependencySubgraphNode& destination) const
	{
		return (m_backEdges.find({source, destination}) != m_backEdges.end());
	}

	const std::unordered_set<const HorseIR::SymbolTable::Symbol *>& GetEdgeData(const DependencySubgraphNode& source, const DependencySubgraphNode& destination) const
	{
		return (m_edgeData.at({source, destination}));
	}

private:
	unsigned int GetLinearOutDegree(const DependencySubgraphNode& node) const override
	{
		auto edges = 0u;
		for (const auto& successor : GetSuccessors(node))
		{
			// Exclude back edges for linear orderings

			if (IsBackEdge(node, successor))
			{
				continue;
			}
			edges++;
		}
		return edges;
	}

	using EdgeType = std::pair<DependencySubgraphNode, DependencySubgraphNode>;

	struct EdgeHash
	{
		inline std::size_t operator()(const EdgeType& pair) const
		{
			return (std::hash<DependencySubgraphNode>()(pair.first) * 31 + std::hash<DependencySubgraphNode>()(pair.second));
		}
	};

	std::unordered_set<EdgeType, EdgeHash> m_backEdges;
	std::unordered_map<EdgeType, std::unordered_set<const HorseIR::SymbolTable::Symbol *>, EdgeHash> m_edgeData;
};

}
