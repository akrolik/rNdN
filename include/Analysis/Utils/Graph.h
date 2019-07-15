#pragma once

#include <unordered_map>
#include <unordered_set>

namespace Analysis {

template<typename T>
class Graph
{
public:
	void InsertNode(const T *node)
	{
		// Create if needed

		m_outgoingEdges[node];
		m_incomingEdges[node];

		m_nodes.insert(node);
	}

	void InsertEdge(const T *source, const T *destination)
	{
		m_outgoingEdges[source].insert(destination);
		m_incomingEdges[destination].insert(source);
	}

	const std::unordered_set<const T *>& GetNodes() const { return m_nodes; }
	bool ContainsNode(const T *node) const { return (m_nodes.find(node) != m_nodes.end()); }

	const std::unordered_set<const T *>& GetIncomingEdges(const T *node) const { return m_incomingEdges.at(node); }
	const std::unordered_set<const T *>& GetOutgoingEdges(const T *node) const { return m_outgoingEdges.at(node); }

	unsigned int GetInDegree(const T *node) const
	{
		if (m_incomingEdges.find(node) == m_incomingEdges.end())
		{
			return 0;
		}
		return m_incomingEdges.at(node).size();
	}

	unsigned int GetOutDegree(const T *node) const
	{
		if (m_outgoingEdges.find(node) == m_outgoingEdges.end())
		{
			return 0;
		}
		return m_outgoingEdges.at(node).size();
	}

protected:
	std::unordered_set<const T *> m_nodes;

	std::unordered_map<const T *, std::unordered_set<const T *>> m_outgoingEdges;
	std::unordered_map<const T *, std::unordered_set<const T *>> m_incomingEdges;
};

}
