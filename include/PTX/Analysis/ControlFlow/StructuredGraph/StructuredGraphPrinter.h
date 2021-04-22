#pragma once

#include <string>

#include "PTX/Analysis/ControlFlow/StructuredGraph/ConstStructuredGraphVisitor.h"
#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraph.h"

#include "Libraries/robin_hood.h"

namespace PTX {
namespace Analysis {

class StructuredGraphPrinter : public ConstStructuredGraphVisitor
{
public:
	static std::string PrettyString(const std::string& name, const StructureNode *node);

	void Visit(const StructureNode *node) override;

	void Visit(const BranchStructure *structure) override;
	void Visit(const ExitStructure *structure) override;
	void Visit(const LoopStructure *structure) override;
	void Visit(const SequenceStructure *structure) override;

private:
	void Indent();
	unsigned int m_indent = 0;
	unsigned int m_index = 0;
	std::stringstream m_string;

	robin_hood::unordered_map<const StructureNode *, std::string> m_nameMap;
};

}
}
