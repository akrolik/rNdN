#pragma once

#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "HorseIR/Analysis/Dependency/DependencyAccessAnalysis.h"
#include "HorseIR/Analysis/Dependency/DependencyGraph.h"
#include "HorseIR/Analysis/Dependency/Overlay/DependencyOverlay.h"
#include "HorseIR/Analysis/Helpers/GPUAnalysisHelper.h"

namespace HorseIR {
namespace Analysis {

class DependencyAnalysis : public ConstHierarchicalVisitor
{
public:
	DependencyAnalysis(const DependencyAccessAnalysis& accessAnalysis) : m_accessAnalysis(accessAnalysis) {}

	// Analysis inputs and outputs

	void Build(const Function *function);

	const DependencyGraph *GetGraph() const { return m_graph; }
	FunctionDependencyOverlay *GetOverlay() const { return m_functionOverlay; }

	// Function

	bool VisitIn(const Function *function) override;
	void VisitOut(const Function *function) override;

	// Statements

	bool VisitIn(const Statement *statement) override;
	void VisitOut(const Statement *statement) override;

	template<typename T>
	void VisitCompoundStatement(const typename T::NodeType *statement);

	bool VisitIn(const IfStatement *ifS) override;
	bool VisitIn(const WhileStatement *whileS) override;
	bool VisitIn(const RepeatStatement *repeatS) override;

	void VisitOut(const IfStatement *ifS) override;
	void VisitOut(const WhileStatement *whileS) override;
	void VisitOut(const RepeatStatement *repeatS) override;

	bool VisitIn(const BlockStatement *blockS) override;
	void VisitOut(const BlockStatement *blockS) override;

	bool VisitIn(const AssignStatement *assignS) override;

	// Expressions

	void VisitOut(const Expression *expression) override;

	bool VisitIn(const FunctionLiteral *literal) override;
	void VisitOut(const FunctionLiteral *literal) override;

	bool VisitIn(const Identifier *identifier) override;

private:
	const DependencyAccessAnalysis& m_accessAnalysis;
	const Statement *m_currentStatement = nullptr;

	DependencyGraph *m_graph = new DependencyGraph();
	DependencyOverlay *m_graphOverlay = nullptr;
	FunctionDependencyOverlay *m_functionOverlay = nullptr;

	GPUAnalysisHelper m_gpuHelper;
	bool m_isTarget = false;
	unsigned int m_index = 0;
};

}
}
