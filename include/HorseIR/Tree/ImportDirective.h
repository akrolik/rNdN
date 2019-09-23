#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/ModuleContent.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class ImportDirective : public ModuleContent
{
public:
	ImportDirective(const std::string& module, const std::string& contents) : m_module(module), m_contents({contents}) {}
	ImportDirective(const std::string& module, const std::vector<std::string>& contents) : m_module(module), m_contents(contents) {}

	ImportDirective *Clone() const override
	{
		return new ImportDirective(m_module, m_contents);
	}

	const std::string& GetModuleName() const { return m_module; }
	void SetModuleName(const std::string& module) { m_module = module; }

	const std::vector<std::string>& GetContents() const { return m_contents; }
	void SetContents(const std::vector<std::string>& contents) { m_contents = contents; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

protected:
	std::string m_module;
	std::vector<std::string> m_contents;
};

}
