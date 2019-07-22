#pragma once

#include <string_view>
#include <unordered_set>
#include <vector>

#include "Analysis/Compatibility/CompatibilityGraph.h"

#include "Analysis/Compatibility/Overlay/CompatibilityOverlayVisitor.h"
#include "Analysis/Compatibility/Overlay/CompatibilityOverlayConstVisitor.h"

#include "HorseIR/Tree/Tree.h"

using namespace std::literals::string_view_literals;

namespace Analysis {

class CompatibilityOverlay
{
public:
	CompatibilityOverlay(const CompatibilityGraph *graph, CompatibilityOverlay *parent = nullptr) : m_graph(graph), m_parent(parent)
	{
		if (m_parent != nullptr)
		{
			m_parent->m_children.push_back(this);
		}
	}

	virtual std::string_view GetName() const { return ""sv; }

	const CompatibilityGraph *GetGraph() const { return m_graph; }

	CompatibilityOverlay *GetParent() const { return m_parent; }
	void SetParent(CompatibilityOverlay *parent, bool update = true)
	{
		m_parent = parent;
		if (m_parent != nullptr)
		{
			m_parent->m_children.push_back(this);
		}
	}

	const std::vector<CompatibilityOverlay *>& GetChildren() const { return m_children; }
	CompatibilityOverlay *GetChild(unsigned int index) const { return m_children.at(index); }

	void SetChildren(const std::vector<CompatibilityOverlay *>& children, bool update = true)
	{
		m_children = children;
		for (auto child : children)
		{
			child->m_parent = this;
		}
	}
	void AddChild(CompatibilityOverlay *child, bool update = true)
	{
		m_children.push_back(child);
		child->m_parent = this;
	}

	const std::unordered_set<const HorseIR::Statement *>& GetStatements() const { return m_statements; }
	void SetStatements(const std::unordered_set<const HorseIR::Statement *>& statements) { m_statements = statements; }
	void InsertStatement(const HorseIR::Statement *statement) { m_statements.insert(statement); }

	bool ContainsStatement(const HorseIR::Statement *statement) const { return m_statements.find(statement) != m_statements.end(); }

	bool IsReducible() const
	{
		return (m_children.size() == 1 && m_statements.size() == 0);
	}

	virtual bool IsGPU() const { return false; }
	bool IsSynchronized() const
	{
		for (const auto statement : m_statements)
		{
			if (m_graph->IsSynchronizedNode(statement))
			{
				return true;
			}
		}

		for (const auto child : m_children)
		{
			if (child->IsSynchronized())
			{
				return true;
			}
		}

		return false;
	}

	virtual void Accept(CompatibilityOverlayVisitor& visitor) { visitor.Visit(this); }
	virtual void Accept(CompatibilityOverlayConstVisitor& visitor) const { visitor.Visit(this); }

protected:
	const CompatibilityGraph *m_graph = nullptr;

	CompatibilityOverlay *m_parent = nullptr;
	std::vector<CompatibilityOverlay *> m_children;

	std::unordered_set<const HorseIR::Statement *> m_statements;
};

class KernelCompatibilityOverlay : public CompatibilityOverlay
{
public:
	using CompatibilityOverlay::CompatibilityOverlay;

	std::string_view GetName() const override { return "Kernel"sv; }

	bool IsGPU() const { return true; }

	void Accept(CompatibilityOverlayVisitor& visitor) { visitor.Visit(this); }
	void Accept(CompatibilityOverlayConstVisitor& visitor) const { visitor.Visit(this); }
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

	std::string_view GetName() const override { return m_node->GetName(); }

	CompatibilityOverlay *GetBody() const { return m_children.at(0); }

	void Accept(CompatibilityOverlayVisitor& visitor) { visitor.Visit(this); }
	void Accept(CompatibilityOverlayConstVisitor& visitor) const { visitor.Visit(this); }
};

class IfCompatibilityOverlay : public CompoundCompatibilityOverlay<HorseIR::IfStatement>
{
public:
	using CompoundCompatibilityOverlay<HorseIR::IfStatement>::CompoundCompatibilityOverlay;

	std::string_view GetName() const override { return "If"sv; }

	CompatibilityOverlay *GetTrueBranch() const { return m_children.at(0); }
	CompatibilityOverlay *GetElseBranch() const { return m_children.at(1); }

	void Accept(CompatibilityOverlayVisitor& visitor) { visitor.Visit(this); }
	void Accept(CompatibilityOverlayConstVisitor& visitor) const { visitor.Visit(this); }
};

class WhileCompatibilityOverlay : public CompoundCompatibilityOverlay<HorseIR::WhileStatement>
{
public:
	using CompoundCompatibilityOverlay<HorseIR::WhileStatement>::CompoundCompatibilityOverlay;

	std::string_view GetName() const override { return "While"sv; }

	CompatibilityOverlay *GetBody() const { return m_children.at(0); }

	void Accept(CompatibilityOverlayVisitor& visitor) { visitor.Visit(this); }
	void Accept(CompatibilityOverlayConstVisitor& visitor) const { visitor.Visit(this); }
};

class RepeatCompatibilityOverlay : public CompoundCompatibilityOverlay<HorseIR::RepeatStatement>
{
public:
	using CompoundCompatibilityOverlay<HorseIR::RepeatStatement>::CompoundCompatibilityOverlay;

	std::string_view GetName() const override { return "Repeat"sv; }

	CompatibilityOverlay *GetBody() const { return m_children.at(0); }

	void Accept(CompatibilityOverlayVisitor& visitor) { visitor.Visit(this); }
	void Accept(CompatibilityOverlayConstVisitor& visitor) const { visitor.Visit(this); }
};

}
