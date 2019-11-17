#pragma once

#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "Analysis/Dependency/DependencyAccessAnalysis.h"
#include "Analysis/Dependency/DependencyGraph.h"
#include "Analysis/Dependency/Overlay/DependencyOverlay.h"

#include "Analysis/Helpers/GPUAnalysisHelper.h"

namespace Analysis {

class DependencyAnalysis : public HorseIR::ConstHierarchicalVisitor
{
public:
	DependencyAnalysis(const DependencyAccessAnalysis& accessAnalysis) : m_accessAnalysis(accessAnalysis) {}

	// Analysis inputs and outputs

	void Build(const HorseIR::Function *function);

	const DependencyGraph *GetGraph() const { return m_graph; }
	FunctionDependencyOverlay *GetOverlay() const { return m_functionOverlay; }

	// Function

	bool VisitIn(const HorseIR::Function *function) override;
	void VisitOut(const HorseIR::Function *function) override;

	// Statements

	bool VisitIn(const HorseIR::Statement *statement) override;
	void VisitOut(const HorseIR::Statement *statement) override;

	template<typename T>
	void VisitCompoundStatement(const typename T::NodeType *statement);

	bool VisitIn(const HorseIR::IfStatement *ifS) override;
	bool VisitIn(const HorseIR::WhileStatement *whileS) override;
	bool VisitIn(const HorseIR::RepeatStatement *repeatS) override;

	void VisitOut(const HorseIR::IfStatement *ifS) override;
	void VisitOut(const HorseIR::WhileStatement *whileS) override;
	void VisitOut(const HorseIR::RepeatStatement *repeatS) override;

	bool VisitIn(const HorseIR::BlockStatement *blockS) override;
	void VisitOut(const HorseIR::BlockStatement *blockS) override;

	bool VisitIn(const HorseIR::AssignStatement *assignS) override;

	// Expressions

	void VisitOut(const HorseIR::Expression *expression) override;

	bool VisitIn(const HorseIR::FunctionLiteral *literal) override;
	void VisitOut(const HorseIR::FunctionLiteral *literal) override;

	bool VisitIn(const HorseIR::Identifier *identifier) override;

private:
	const DependencyAccessAnalysis& m_accessAnalysis;
	const HorseIR::Statement *m_currentStatement = nullptr;

	DependencyGraph *m_graph = new DependencyGraph();
	DependencyOverlay *m_graphOverlay = nullptr;
	FunctionDependencyOverlay *m_functionOverlay = nullptr;

	GPUAnalysisHelper m_gpuHelper;
	bool m_isTarget = false;
	unsigned int m_index = 0;
};

}
