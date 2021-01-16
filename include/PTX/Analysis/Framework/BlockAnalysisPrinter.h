#pragma once

#include "PTX/Traversal/ConstHierarchicalVisitor.h"

#include <string>
#include <sstream>

#include "PTX/Analysis/Framework/BlockAnalysis.h"
#include "PTX/Tree/Tree.h"

namespace PTX {
namespace Analysis {

class BlockAnalysisPrinter : public ConstHierarchicalVisitor
{
public:
	static std::string PrettyString(const BlockAnalysis& analysis, const FunctionDefinition<VoidType> *function);

	BlockAnalysisPrinter(const BlockAnalysis& analysis) : m_analysis(analysis) {}

	bool VisitIn(const BasicBlock *block) override;
	bool VisitIn(const Statement *statement) override;

protected:
	std::string Indent() const;

	std::stringstream m_string;
	unsigned int m_indent = 0;

	const BlockAnalysis& m_analysis;
};

}
}
