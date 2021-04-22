#pragma once

#include <algorithm>
#include <string_view>
#include <vector>

#include "HorseIR/Analysis/Dependency/DependencyGraph.h"
#include "HorseIR/Analysis/Dependency/Overlay/DependencyOverlayVisitor.h"
#include "HorseIR/Analysis/Dependency/Overlay/DependencyOverlayConstVisitor.h"

#include "HorseIR/Tree/Tree.h"

#include "Libraries/robin_hood.h"

using namespace std::literals::string_view_literals;

namespace HorseIR {
namespace Analysis {

class DependencyOverlay
{
public:
	DependencyOverlay(const DependencyGraph *graph, DependencyOverlay *parent = nullptr) : m_graph(graph), m_parent(parent)
	{
		if (m_parent != nullptr)
		{
			m_parent->m_children.push_back(this);
		}
	}

	virtual std::string_view GetName() const { return ""sv; }

	const DependencyGraph *GetGraph() const { return m_graph; }

	// Subgraph

	const DependencySubgraph *GetSubgraph() const { return m_subgraph; }
	DependencySubgraph *GetSubgraph() { return m_subgraph; }
	void SetSubgraph(DependencySubgraph *subgraph) { m_subgraph = subgraph; }

	// Parent

	DependencyOverlay *GetParent() const { return m_parent; }
	void SetParent(DependencyOverlay *parent)
	{
		m_parent = parent;
		if (m_parent != nullptr)
		{
			m_parent->m_children.push_back(this);
		}
	}

	// Children

	std::vector<const DependencyOverlay *> GetChildren() const
	{
		return { std::begin(m_children), std::end(m_children) };
	}
	std::vector<DependencyOverlay *>& GetChildren() { return m_children; }

	const DependencyOverlay *GetChild(unsigned int index) const { return m_children.at(index); }
	DependencyOverlay *GetChild(unsigned int index) { return m_children.at(index); }

	void SetChildren(const std::vector<DependencyOverlay *>& children)
	{
		m_children = children;
		for (auto child : children)
		{
			child->m_parent = this;
		}
	}
	void AddChild(DependencyOverlay *child)
	{
		if (std::find(m_children.begin(), m_children.end(), child) == m_children.end())
		{
			m_children.push_back(child);
			child->m_parent = this;
		}
	}
	void RemoveChild(const DependencyOverlay *child)
	{
		m_children.erase(std::remove(m_children.begin(), m_children.end(), child), m_children.end());
	}

	// Statements

	const robin_hood::unordered_set<const Statement *>& GetStatements() const { return m_statements; }
	void InsertStatement(const Statement *statement) { m_statements.insert(statement); }

	bool ContainsStatement(const Statement *statement) const { return m_statements.find(statement) != m_statements.end(); }

	// Properties

	bool IsReducible() const
	{
		return (m_children.size() == 1 && m_children.at(0)->IsGPU() && m_statements.size() == 0);
	}

	bool IsGPU() const { return m_gpu; }
	void SetGPU(bool gpu) { m_gpu = gpu; }

	bool IsSynchronizedOut() const { return m_synchronizedOut; }
	void SetSynchronizedOut(bool synchronizedOut) { m_synchronizedOut = synchronizedOut; }

	// Visitors

	virtual void Accept(DependencyOverlayVisitor& visitor) { visitor.Visit(this); }
	virtual void Accept(DependencyOverlayConstVisitor& visitor) const { visitor.Visit(this); }

protected:
	const DependencyGraph *m_graph = nullptr;
	DependencySubgraph *m_subgraph = nullptr;

	DependencyOverlay *m_parent = nullptr;
	std::vector<DependencyOverlay *> m_children;

	robin_hood::unordered_set<const Statement *> m_statements;

	bool m_gpu = false;
	bool m_synchronizedOut = false;
};

template<typename T>
class CompoundDependencyOverlay : public DependencyOverlay
{
public:
	using NodeType = T;

	CompoundDependencyOverlay(const T *node, const DependencyGraph *graph, DependencyOverlay *parent = nullptr) : DependencyOverlay(graph, parent), m_node(node) {}

	const T *GetNode() const { return m_node; }

protected:
	const T *m_node = nullptr;
};

class FunctionDependencyOverlay : public CompoundDependencyOverlay<Function>
{
public:
	using CompoundDependencyOverlay<Function>::CompoundDependencyOverlay;

	std::string_view GetName() const override { return m_node->GetName(); }

	const DependencyOverlay *GetBody() const { return m_children.at(0); }
	DependencyOverlay *GetBody() { return m_children.at(0); }

	void Accept(DependencyOverlayVisitor& visitor) { visitor.Visit(this); }
	void Accept(DependencyOverlayConstVisitor& visitor) const { visitor.Visit(this); }
};

class IfDependencyOverlay : public CompoundDependencyOverlay<IfStatement>
{
public:
	using CompoundDependencyOverlay<IfStatement>::CompoundDependencyOverlay;

	std::string_view GetName() const override { return "If"sv; }

	const DependencyOverlay *GetTrueBranch() const { return m_children.at(0); }
	DependencyOverlay *GetTrueBranch() { return m_children.at(0); }

	const DependencyOverlay *GetElseBranch() const { return m_children.at(1); }
	DependencyOverlay *GetElseBranch() { return m_children.at(1); }

	bool HasElseBranch() const { return (m_children.size() > 1); }

	void Accept(DependencyOverlayVisitor& visitor) { visitor.Visit(this); }
	void Accept(DependencyOverlayConstVisitor& visitor) const { visitor.Visit(this); }
};

class WhileDependencyOverlay : public CompoundDependencyOverlay<WhileStatement>
{
public:
	using CompoundDependencyOverlay<WhileStatement>::CompoundDependencyOverlay;

	std::string_view GetName() const override { return "While"sv; }

	const DependencyOverlay *GetBody() const { return m_children.at(0); }
	DependencyOverlay *GetBody() { return m_children.at(0); }

	void Accept(DependencyOverlayVisitor& visitor) { visitor.Visit(this); }
	void Accept(DependencyOverlayConstVisitor& visitor) const { visitor.Visit(this); }
};

class RepeatDependencyOverlay : public CompoundDependencyOverlay<RepeatStatement>
{
public:
	using CompoundDependencyOverlay<RepeatStatement>::CompoundDependencyOverlay;

	std::string_view GetName() const override { return "Repeat"sv; }

	const DependencyOverlay *GetBody() const { return m_children.at(0); }
	DependencyOverlay *GetBody() { return m_children.at(0); }

	void Accept(DependencyOverlayVisitor& visitor) { visitor.Visit(this); }
	void Accept(DependencyOverlayConstVisitor& visitor) const { visitor.Visit(this); }
};

}
}
