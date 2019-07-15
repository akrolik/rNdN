#pragma once

#include <unordered_set>

#include "Analysis/Compatibility/CompatibilityGraph.h"

#include "Analysis/Compatibility/Overlay/CompatibilityOverlayVisitor.h"
#include "Analysis/Compatibility/Overlay/CompatibilityOverlayConstVisitor.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

class CompatibilityOverlay
{
public:
	CompatibilityOverlay(const CompatibilityGraph *graph, CompatibilityOverlay *parent = nullptr) : m_graph(graph), m_parent(parent)
{
	if (m_parent != nullptr)
	{
		m_parent->m_children.insert(this);
	}
}

	const CompatibilityGraph *GetGraph() const { return m_graph; }

	CompatibilityOverlay *GetParent() const { return m_parent; }
	void SetParent(CompatibilityOverlay *parent) { m_parent = parent; }

	const std::unordered_set<CompatibilityOverlay *>& GetChildren() const { return m_children; }
	void SetChildren(const std::unordered_set<CompatibilityOverlay *>& children) { m_children = children; }

	void InsertStatement(const HorseIR::Statement *statement) { m_statements.insert(statement); }
	const std::unordered_set<const HorseIR::Statement *>& GetStatements() const { return m_statements; }

	bool ContainsStatement(const HorseIR::Statement *statement) const { return m_statements.find(statement) != m_statements.end(); }

	virtual void Accept(CompatibilityOverlayVisitor& visitor) { visitor.Visit(this); }
	virtual void Accept(CompatibilityOverlayConstVisitor& visitor) const { visitor.Visit(this); }

protected:
	const CompatibilityGraph *m_graph = nullptr;

	CompatibilityOverlay *m_parent = nullptr;
	std::unordered_set<CompatibilityOverlay *> m_children;

	std::unordered_set<const HorseIR::Statement *> m_statements;
};

template<typename T>
class CompoundCompatibilityOverlay : public CompatibilityOverlay
{
public:
	using NodeType = T;

	CompoundCompatibilityOverlay(const T *node, const CompatibilityGraph *graph, CompatibilityOverlay *parent = nullptr) : CompatibilityOverlay(graph, parent), m_node(node) {}

	const T *GetNode() const { return m_node; }


protected:
	const T *m_node = nullptr;
};

class FunctionCompatibilityOverlay : public CompoundCompatibilityOverlay<HorseIR::Function>
{
public:
	using CompoundCompatibilityOverlay<HorseIR::Function>::CompoundCompatibilityOverlay;

	void Accept(CompatibilityOverlayVisitor& visitor) { visitor.Visit(this); }
	void Accept(CompatibilityOverlayConstVisitor& visitor) const { visitor.Visit(this); }
};

class KernelCompatibilityOverlay : public CompatibilityOverlay
{
public:
	using CompatibilityOverlay::CompatibilityOverlay;

	void Accept(CompatibilityOverlayVisitor& visitor) { visitor.Visit(this); }
	void Accept(CompatibilityOverlayConstVisitor& visitor) const { visitor.Visit(this); }
};

class IfCompatibilityOverlay : public CompoundCompatibilityOverlay<HorseIR::IfStatement>
{
public:
	using CompoundCompatibilityOverlay<HorseIR::IfStatement>::CompoundCompatibilityOverlay;

	void Accept(CompatibilityOverlayVisitor& visitor) { visitor.Visit(this); }
	void Accept(CompatibilityOverlayConstVisitor& visitor) const { visitor.Visit(this); }
};

class WhileCompatibilityOverlay : public CompoundCompatibilityOverlay<HorseIR::WhileStatement>
{
public:
	using CompoundCompatibilityOverlay<HorseIR::WhileStatement>::CompoundCompatibilityOverlay;

	void Accept(CompatibilityOverlayVisitor& visitor) { visitor.Visit(this); }
	void Accept(CompatibilityOverlayConstVisitor& visitor) const { visitor.Visit(this); }
};

class RepeatCompatibilityOverlay : public CompoundCompatibilityOverlay<HorseIR::RepeatStatement>
{
public:
	using CompoundCompatibilityOverlay<HorseIR::RepeatStatement>::CompoundCompatibilityOverlay;

	void Accept(CompatibilityOverlayVisitor& visitor) { visitor.Visit(this); }
	void Accept(CompatibilityOverlayConstVisitor& visitor) const { visitor.Visit(this); }
};

}
