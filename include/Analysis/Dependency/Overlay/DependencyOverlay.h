#pragma once

#include <algorithm>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "Analysis/Dependency/DependencyGraph.h"
#include "Analysis/Dependency/Overlay/DependencyOverlayVisitor.h"
#include "Analysis/Dependency/Overlay/DependencyOverlayConstVisitor.h"

#include "HorseIR/Tree/Tree.h"

using namespace std::literals::string_view_literals;

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

	DependencySubgraph *GetSubgraph() const { return m_subgraph; }
	void SetSubgraph(DependencySubgraph *subgraph) { m_subgraph = subgraph; }

	DependencyOverlay *GetParent() const { return m_parent; }
	void SetParent(DependencyOverlay *parent)
	{
		m_parent = parent;
		if (m_parent != nullptr)
		{
			m_parent->m_children.push_back(this);
		}
	}

	const std::vector<DependencyOverlay *>& GetChildren() const { return m_children; }
	DependencyOverlay *GetChild(unsigned int index) const { return m_children.at(index); }

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
		m_children.push_back(child);
		child ->m_parent = this;
	}
	void RemoveChild(DependencyOverlay *child)
	{
		m_children.erase(std::remove(m_children.begin(), m_children.end(), child), m_children.end());
	}

	const std::unordered_set<const HorseIR::Statement *>& GetStatements() const { return m_statements; }
	void InsertStatement(const HorseIR::Statement *statement) { m_statements.insert(statement); }

	bool ContainsStatement(const HorseIR::Statement *statement) const { return m_statements.find(statement) != m_statements.end(); }

	bool IsReducible() const
	{
		return (m_children.size() == 1 && m_children.at(0)->IsGPU() && m_statements.size() == 0);
	}

	bool IsGPU() const { return m_gpu; }
	void SetGPU(bool gpu) { m_gpu = gpu; }

	bool IsSynchronizedOut() const { return m_synchronizedOut; }
	void SetSynchronizedOut(bool synchronizedOut) { m_synchronizedOut = synchronizedOut; }

	virtual void Accept(DependencyOverlayVisitor& visitor) { visitor.Visit(this); }
	virtual void Accept(DependencyOverlayConstVisitor& visitor) const { visitor.Visit(this); }

protected:
	const DependencyGraph *m_graph = nullptr;
	DependencySubgraph *m_subgraph = nullptr;

	DependencyOverlay *m_parent = nullptr;
	std::vector<DependencyOverlay *> m_children;

	std::unordered_set<const HorseIR::Statement *> m_statements;

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

class FunctionDependencyOverlay : public CompoundDependencyOverlay<HorseIR::Function>
{
public:
	using CompoundDependencyOverlay<HorseIR::Function>::CompoundDependencyOverlay;

	std::string_view GetName() const override { return m_node->GetName(); }

	DependencyOverlay *GetBody() const { return m_children.at(0); }

	void Accept(DependencyOverlayVisitor& visitor) { visitor.Visit(this); }
	void Accept(DependencyOverlayConstVisitor& visitor) const { visitor.Visit(this); }
};

class IfDependencyOverlay : public CompoundDependencyOverlay<HorseIR::IfStatement>
{
public:
	using CompoundDependencyOverlay<HorseIR::IfStatement>::CompoundDependencyOverlay;

	std::string_view GetName() const override { return "If"sv; }

	DependencyOverlay *GetTrueBranch() const { return m_children.at(0); }
	DependencyOverlay *GetElseBranch() const { return m_children.at(1); }

	bool HasElseBranch() const { return (m_children.size() > 1); }

	void Accept(DependencyOverlayVisitor& visitor) { visitor.Visit(this); }
	void Accept(DependencyOverlayConstVisitor& visitor) const { visitor.Visit(this); }
};

class WhileDependencyOverlay : public CompoundDependencyOverlay<HorseIR::WhileStatement>
{
public:
	using CompoundDependencyOverlay<HorseIR::WhileStatement>::CompoundDependencyOverlay;

	std::string_view GetName() const override { return "While"sv; }

	DependencyOverlay *GetBody() const { return m_children.at(0); }

	void Accept(DependencyOverlayVisitor& visitor) { visitor.Visit(this); }
	void Accept(DependencyOverlayConstVisitor& visitor) const { visitor.Visit(this); }
};

class RepeatDependencyOverlay : public CompoundDependencyOverlay<HorseIR::RepeatStatement>
{
public:
	using CompoundDependencyOverlay<HorseIR::RepeatStatement>::CompoundDependencyOverlay;

	std::string_view GetName() const override { return "Repeat"sv; }

	DependencyOverlay *GetBody() const { return m_children.at(0); }

	void Accept(DependencyOverlayVisitor& visitor) { visitor.Visit(this); }
	void Accept(DependencyOverlayConstVisitor& visitor) const { visitor.Visit(this); }
};

}
