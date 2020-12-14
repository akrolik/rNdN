#pragma once

#include "HorseIR/Tree/Module.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class LibraryModule : public Module
{
public:
	using Module::Module;

	Module *Clone() const override
	{
		std::vector<ModuleContent *> contents;
		for (const auto& content : m_contents)
		{
			contents.push_back(content->Clone());
		}
		return new LibraryModule(m_name, contents);
	}

	// Visitors

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			for (auto& content : m_contents)
			{
				content->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			for (const auto& content : m_contents)
			{
				content->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}
};

}
