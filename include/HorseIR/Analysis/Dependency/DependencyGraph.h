#pragma once

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

#include "Utils/Graph.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {
namespace Analysis {

template<typename N>
class DependencyGraphBase : public Utils::Graph<N>
{
public:
	using Utils::Graph<N>::Graph;

	void InsertNode(const N& node, bool isGPU, bool isLibrary)
	{
		Utils::Graph<N>::InsertNode(node);

		if (isGPU)
		{
			m_gpuNodes.insert(node);
		}
		if (isLibrary)
		{
			m_libraryNodes.insert(node);
		}
	}

	bool IsGPUNode(const N& node) const
	{
		return (m_gpuNodes.find(node) != m_gpuNodes.end());
	}

	bool IsGPULibraryNode(const N& node) const
	{
		return (m_libraryNodes.find(node) != m_libraryNodes.end());
	}

	void RemoveNode(const N& node) override
	{
		m_gpuNodes.erase(node);
		m_libraryNodes.erase(node);

		Utils::Graph<N>::RemoveNode(node);
	}

	void InsertEdge(const N& source, const N& destination, const SymbolTable::Symbol *symbol, bool isBackEdge, bool isSynchronized)
	{
		InsertEdge(source, destination, std::unordered_set<const SymbolTable::Symbol *>({symbol}), isBackEdge, isSynchronized);
	}

	void InsertEdge(const N& source, const N& destination, const std::unordered_set<const SymbolTable::Symbol *>& symbols, bool isBackEdge, bool isSynchronized)
	{
		Utils::Graph<N>::InsertEdge(source, destination);

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

		// Add to set of synchronized edges if needed

		if (isSynchronized)
		{
			m_synchronizedEdges.insert({source, destination});
		}
	}

	const std::unordered_set<const SymbolTable::Symbol *>& GetEdgeData(const N& source, const N& destination) const
	{
		return (m_edgeData.at({source, destination}));
	}

	bool IsBackEdge(const N& source, const N& destination) const
	{
		return (m_backEdges.find({source, destination}) != m_backEdges.end());
	}

	bool IsSynchronizedEdge(const N& source, const N& destination) const
	{
		return (m_synchronizedEdges.find({source, destination}) != m_synchronizedEdges.end());
	}

	void RemoveEdge(const N& source, const N& destination) override
	{
		m_edgeData.erase({source, destination});
		m_backEdges.erase({source, destination});
		m_synchronizedEdges.erase({source, destination});

		Utils::Graph<N>::RemoveEdge(source, destination);
	}

private:
	unsigned int GetLinearInDegree(const N& node) const override
	{
		auto edges = 0u;
		for (const auto& predecessor : this->GetPredecessors(node))
		{
			// Exclude back edges for linear orderings

			if (IsBackEdge(predecessor, node))
			{
				continue;
			}
			edges++;
		}
		return edges;
	}

	unsigned int GetLinearOutDegree(const N& node) const override
	{
		auto edges = 0u;
		for (const auto& successor : this->GetSuccessors(node))
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

	std::unordered_set<N> m_gpuNodes;
	std::unordered_set<N> m_libraryNodes;

	using EdgeType = std::pair<N, N>;
	struct EdgeHash
	{
		inline std::size_t operator()(const EdgeType& pair) const
		{
			return (std::hash<N>()(pair.first) * 31 + std::hash<N>()(pair.second));
		}
	};

	std::unordered_set<EdgeType, EdgeHash> m_backEdges;
	std::unordered_set<EdgeType, EdgeHash> m_synchronizedEdges;
	std::unordered_map<EdgeType, std::unordered_set<const SymbolTable::Symbol *>, EdgeHash> m_edgeData;
};

using DependencyGraphNode = const Statement *;
using DependencyGraph = DependencyGraphBase<DependencyGraphNode>;

class DependencyOverlay;

using DependencySubgraphNode = std::variant<const Statement *, const DependencyOverlay *>;
using DependencySubgraph = DependencyGraphBase<DependencySubgraphNode>;

}
}
