#pragma once

#include <string>
#include <sstream>

#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "HorseIR/Analysis/Framework/StatementAnalysis.h"
#include "HorseIR/Tree/Tree.h"

namespace HorseIR {
namespace Analysis {

class StatementAnalysisPrinter : public ConstHierarchicalVisitor
{
public:
	static std::string PrettyString(const StatementAnalysis& analysis, const Function *function);

	StatementAnalysisPrinter(const StatementAnalysis& analysis) : m_analysis(analysis) {}

	bool VisitIn(const Statement *statement) override;
	bool VisitIn(const IfStatement *ifS) override;
	bool VisitIn(const WhileStatement *whileS) override;
	bool VisitIn(const RepeatStatement *repeatS) override;

	bool VisitIn(const BlockStatement *blockS) override;
	void VisitOut(const BlockStatement *blockS) override;

protected:
	std::string Indent() const;

	std::stringstream m_string;
	unsigned int m_indent = 0;

	const StatementAnalysis& m_analysis;
};

}
}
