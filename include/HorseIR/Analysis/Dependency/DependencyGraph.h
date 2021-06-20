#pragma once

#include <utility>

#include "HorseIR/Analysis/Dependency/Overlay/DependencyOverlay.h"

#include "Utils/Graph.h"
#include "Utils/Variant.h"

#include "HorseIR/Tree/Tree.h"

#include "Libraries/robin_hood.h"

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
		InsertEdge(source, destination, robin_hood::unordered_set<const SymbolTable::Symbol *>({symbol}), isBackEdge, isSynchronized);
	}

	void InsertEdge(const N& source, const N& destination, const robin_hood::unordered_set<const SymbolTable::Symbol *>& symbols, bool isBackEdge, bool isSynchronized)
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

	const robin_hood::unordered_set<const SymbolTable::Symbol *>& GetEdgeData(const N& source, const N& destination) const
	{
		return m_edgeData.at({source, destination});
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

	robin_hood::unordered_set<N> m_gpuNodes;
	robin_hood::unordered_set<N> m_libraryNodes;

	using EdgeType = typename Utils::Graph<N>::EdgeType;
	using EdgeHash = typename Utils::Graph<N>::EdgeHash;

	robin_hood::unordered_set<EdgeType, EdgeHash> m_backEdges;
	robin_hood::unordered_set<EdgeType, EdgeHash> m_synchronizedEdges;
	robin_hood::unordered_map<EdgeType, robin_hood::unordered_set<const SymbolTable::Symbol *>, EdgeHash> m_edgeData;
};

using DependencyGraphNode = const Statement *;
class DependencyGraph : public DependencyGraphBase<DependencyGraphNode>
{
public:
	using DependencyGraphBase<DependencyGraphNode>::DependencyGraphBase;

private:
	std::vector<DependencyGraphNode> OrderNodes(const robin_hood::unordered_set<DependencyGraphNode>& unodes) const override
	{
		std::vector<DependencyGraphNode> nodes(std::begin(unodes), std::end(unodes));
		std::sort(std::begin(nodes), std::end(nodes), [](const Statement *s1, const Statement *s2)
		{
			return s1->GetLineNumber() < s2->GetLineNumber();
		});
		return nodes;
	}
};

using DependencySubgraphNode = std::variant<const Statement *, const DependencyOverlay *>;
class DependencySubgraph : public DependencyGraphBase<DependencySubgraphNode>
{
public:
	using DependencyGraphBase<DependencySubgraphNode>::DependencyGraphBase;

private:
	std::vector<DependencySubgraphNode> OrderNodes(const robin_hood::unordered_set<DependencySubgraphNode>& unodes) const override
	{
		std::vector<DependencySubgraphNode> nodes(std::begin(unodes), std::end(unodes));
		std::sort(std::begin(nodes), std::end(nodes), [](const DependencySubgraphNode& n1, const DependencySubgraphNode& n2)
		{
			auto l1 = std::visit(overloaded
			{
				[&](const Statement *statement)
				{
					return statement->GetLineNumber();
				},
				[&](const Analysis::DependencyOverlay *overlay)
				{
					auto start = std::numeric_limits<int>::max();
					for (auto& statement : overlay->GetStatements())
					{
						if (auto line = statement->GetLineNumber(); line < start)
						{
							start = line;
						}
					}
					return start;
				}},
				n1
			);
			auto l2 = std::visit(overloaded
			{
				[&](const Statement *statement)
				{
					return statement->GetLineNumber();
				},
				[&](const Analysis::DependencyOverlay *overlay)
				{
					auto start = std::numeric_limits<int>::max();
					for (auto& statement : overlay->GetStatements())
					{
						if (auto line = statement->GetLineNumber(); line < start)
						{
							start = line;
						}
					}
					return start;
				}},
				n2
			);
			return l1 < l2;
		});
		return nodes;
	}
};

}
}
