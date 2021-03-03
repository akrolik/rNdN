#pragma once

#include <string>
#include <unordered_map>

#include "PTX/Analysis/ControlFlow/StructuredGraph/ConstStructuredGraphVisitor.h"
#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraph.h"

namespace PTX {
namespace Analysis {

class StructuredGraphPrinter : public ConstStructuredGraphVisitor
{
public:
	static std::string PrettyString(const StructureNode *node);

	void Visit(const StructureNode *node) override;

	void Visit(const BranchStructure *structure) override;
	void Visit(const ExitStructure *structure) override;
	void Visit(const LoopStructure *structure) override;
	void Visit(const SequenceStructure *structure) override;

private:
	void Indent();
	unsigned int m_indent = 0;
	std::stringstream m_string;

	std::unordered_map<const StructureNode *, std::string> m_nameMap;
};

}
}
