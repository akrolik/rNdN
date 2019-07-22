#pragma once

#include <unordered_set>
#include <utility>

#include "Analysis/Utils/Graph.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

class CompatibilityGraph : public Graph<HorseIR::Statement>
{
public:
	using Graph<HorseIR::Statement>::Graph;

	void InsertNode(const HorseIR::Statement *statement, bool isGPU, bool isSynchronized)
	{
		Graph<HorseIR::Statement>::InsertNode(statement);

		// Tag GPU capable nodes

		if (isGPU)
		{
			m_gpuNodes.insert(statement);
		}
		if (isSynchronized)
		{
			m_synchronizedNodes.insert(statement);
		}
	}

	bool IsGPUNode(const HorseIR::Statement *statement) const
	{
		return (m_gpuNodes.find(statement) != m_gpuNodes.end());
	}

	bool IsSynchronizedNode(const HorseIR::Statement *statement) const
	{
		return (m_synchronizedNodes.find(statement) != m_synchronizedNodes.end());
	}

	void InsertEdge(const HorseIR::Statement *source, const HorseIR::Statement *destination, const HorseIR::SymbolTable::Symbol *symbol, bool isBackEdge, bool isCompatible)
	{
		Graph<HorseIR::Statement>::InsertEdge(source, destination);

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

		// Update compatibility with input, must be true for all

		if (m_compatibleEdges.find(edge) == m_compatibleEdges.end())
		{
			m_compatibleEdges[edge] = isCompatible;
		}
		else
		{
			m_compatibleEdges.at(edge) &= isCompatible;
		}

		// Add to set of back edges if needed

		if (isBackEdge)
		{
			m_backEdges.insert({source, destination});
		}
	}

	bool IsBackEdge(const HorseIR::Statement *source, const HorseIR::Statement *destination, bool flipped = false) const
	{
		if (flipped)
		{
			return (m_backEdges.find({destination, source}) != m_backEdges.end());
		}
		return (m_backEdges.find({source, destination}) != m_backEdges.end());
	}

	bool IsCompatibleEdge(const HorseIR::Statement *source, const HorseIR::Statement *destination, bool flipped = false) const
	{
		if (flipped)
		{
			return m_compatibleEdges.at({destination, source});
		}
		return m_compatibleEdges.at({source, destination});
	}

private:
	using EdgeType = std::pair<const HorseIR::Statement *, const HorseIR::Statement *>;

	struct EdgeHash
	{
		inline std::size_t operator()(const EdgeType& pair) const
		{
			return (std::hash<const HorseIR::Statement *>()(pair.first) * 31 + std::hash<const HorseIR::Statement *>()(pair.second));
		}
	};

	std::unordered_set<EdgeType, EdgeHash> m_backEdges;
	std::unordered_map<EdgeType, bool, EdgeHash> m_compatibleEdges;

	std::unordered_set<const HorseIR::Statement *> m_gpuNodes;
	std::unordered_set<const HorseIR::Statement *> m_synchronizedNodes;

	std::unordered_map<EdgeType, std::unordered_set<const HorseIR::SymbolTable::Symbol *>, EdgeHash> m_edgeData;
};

}
