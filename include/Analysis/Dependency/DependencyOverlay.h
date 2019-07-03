#pragma once

#include <unordered_set>

#include "Analysis/Dependency/DependencyOverlayVisitor.h"
#include "Analysis/Dependency/DependencyOverlayConstVisitor.h"
#include "Analysis/Utils/Graph.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

using DependencyGraph = Graph<HorseIR::Statement>;

class DependencyOverlay
{
public:
	DependencyOverlay(const DependencyGraph *graph, DependencyOverlay *parent = nullptr) : m_graph(graph), m_parent(parent)
	{
		if (m_parent != nullptr)
		{
			m_parent->m_children.insert(this);
		}
	}
	
	const DependencyGraph *GetGraph() const { return m_graph; }

	DependencyOverlay *GetParent() const { return m_parent; }
	void SetParent(DependencyOverlay *parent) { m_parent = parent; }

	const std::unordered_set<DependencyOverlay *>& GetChildren() const { return m_children; }

	void InsertStatement(const HorseIR::Statement *statement) { m_statements.insert(statement); }
	const std::unordered_set<const HorseIR::Statement *>& GetStatements() const { return m_statements; }

	virtual void Accept(DependencyOverlayVisitor& visitor) { visitor.Visit(this); }
	virtual void Accept(DependencyOverlayConstVisitor& visitor) const { visitor.Visit(this); }

protected:
	const DependencyGraph *m_graph = nullptr;

	DependencyOverlay *m_parent = nullptr;
	std::unordered_set<DependencyOverlay *> m_children;

	std::unordered_set<const HorseIR::Statement *> m_statements;
};

template<typename T>
class CompoundDependencyOverlay : public DependencyOverlay
{
public:
	CompoundDependencyOverlay(const T *node, const DependencyGraph *graph, DependencyOverlay *parent = nullptr) : DependencyOverlay(graph, parent), m_node(node) {}

	const T *GetNode() const { return m_node; }

	void Accept(DependencyOverlayVisitor& visitor) { visitor.Visit(this); }
	void Accept(DependencyOverlayConstVisitor& visitor) const { visitor.Visit(this); }

protected:
	const T *m_node = nullptr;
};

using FunctionDependencyOverlay = CompoundDependencyOverlay<HorseIR::Function>;
using IfDependencyOverlay = CompoundDependencyOverlay<HorseIR::IfStatement>;
using WhileDependencyOverlay = CompoundDependencyOverlay<HorseIR::WhileStatement>;
using RepeatDependencyOverlay = CompoundDependencyOverlay<HorseIR::RepeatStatement>;

}
