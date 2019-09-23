#pragma once

#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "Analysis/Utils/Graph.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

class DependencyGraph : public Graph<const HorseIR::Statement *>
{
public:
	using NodeType = const HorseIR::Statement *;

	using Graph<const HorseIR::Statement *>::Graph;

	void InsertEdge(const NodeType& source, const NodeType& destination, const HorseIR::SymbolTable::Symbol *symbol, bool isBackEdge)
	{
		Graph<const HorseIR::Statement *>::InsertEdge(source, destination);

		// Add symbol to edge, extending the set if it already exists

		auto edge = std::make_pair(source, destination);
		if (m_edgeData.find(edge) == m_edgeData.end())
		{
			m_edgeData[edge] = {symbol};
		}
		else
		{
			m_edgeData.at(edge).insert(symbol);
		}

		// Add to set of back edges if needed

		if (isBackEdge)
		{
			m_backEdges.insert({source, destination});
		}
	}

	const std::unordered_set<const HorseIR::SymbolTable::Symbol *>& GetEdgeData(const NodeType& source, const NodeType& destination) const
	{
		return (m_edgeData.at({source, destination}));
	}

	bool IsBackEdge(const NodeType& source, const NodeType& destination) const
	{
		return (m_backEdges.find({source, destination}) != m_backEdges.end());
	}

private:
	unsigned int GetLinearOutDegree(const NodeType& node) const override
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

	using EdgeType = std::pair<const HorseIR::Statement *, const HorseIR::Statement *>;

	struct EdgeHash
	{
		inline std::size_t operator()(const EdgeType& pair) const
		{
			return (std::hash<const HorseIR::Statement *>()(pair.first) * 31 + std::hash<const HorseIR::Statement *>()(pair.second));
		}
	};

	std::unordered_set<EdgeType, EdgeHash> m_backEdges;
	std::unordered_map<EdgeType, std::unordered_set<const HorseIR::SymbolTable::Symbol *>, EdgeHash> m_edgeData;
};

}
