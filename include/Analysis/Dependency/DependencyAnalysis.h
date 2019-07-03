#pragma once

#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "Analysis/BasicFlow/UDDUChainsBuilder.h"
#include "Analysis/Dependency/DependencyOverlay.h"

namespace Analysis {

class DependencyAnalysis : public HorseIR::ConstHierarchicalVisitor
{
public:
	DependencyAnalysis(const UDDUChainsBuilder& useDefChains) : m_useDefChains(useDefChains) {}

	void Analyze(const HorseIR::Function *function);
	const DependencyGraph *GetDependencyGraph() const { return m_graph; }
	const DependencyOverlay *GetDependencyOverlay() const { return m_graphOverlay; }

	bool VisitIn(const HorseIR::Function *function) override;

	bool VisitIn(const HorseIR::Statement *statement) override;
	void VisitOut(const HorseIR::Statement *statement) override;

	// Special statements

	bool VisitIn(const HorseIR::AssignStatement *assignS) override;

	template<typename T>
	void VisitCompoundStatement(const T *statement);

	bool VisitIn(const HorseIR::IfStatement *ifS) override;
	bool VisitIn(const HorseIR::WhileStatement *whileS) override;
	bool VisitIn(const HorseIR::RepeatStatement *repeatS) override;

	void VisitOut(const HorseIR::IfStatement *ifS) override;
	void VisitOut(const HorseIR::WhileStatement *whileS) override;
	void VisitOut(const HorseIR::RepeatStatement *repeatS) override;

	bool VisitIn(const HorseIR::BlockStatement *blockS) override;
	void VisitOut(const HorseIR::BlockStatement *blockS) override;

	// Expressions

	bool VisitIn(const HorseIR::FunctionLiteral *literal) override;
	bool VisitIn(const HorseIR::Identifier *identifier) override;

private:
	const UDDUChainsBuilder& m_useDefChains;
	const HorseIR::Statement *m_currentStatement = nullptr;

	DependencyGraph *m_graph = new DependencyGraph();
	DependencyOverlay *m_graphOverlay = nullptr;
};

}
