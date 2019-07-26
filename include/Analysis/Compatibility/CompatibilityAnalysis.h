#pragma once

#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "Analysis/BasicFlow/UDDUChainsBuilder.h"
#include "Analysis/Compatibility/Geometry/GeometryAnalysis.h"

#include "Analysis/Compatibility/CompatibilityGraph.h"
#include "Analysis/Compatibility/Overlay/CompatibilityOverlay.h"

namespace Analysis {

class CompatibilityAnalysis : public HorseIR::ConstHierarchicalVisitor
{
public:
	CompatibilityAnalysis(const UDDUChainsBuilder& useDefChains, const GeometryAnalysis& geometryAnalysis) : m_useDefChains(useDefChains), m_geometryAnalysis(geometryAnalysis) {}

	// Analysis inputs and outputs

	void Analyze(const HorseIR::Function *function);

	const CompatibilityGraph *GetGraph() const { return m_graph; }
	const CompatibilityOverlay *GetOverlay() const { return m_graphOverlay; }

	// Function

	bool VisitIn(const HorseIR::Function *function) override;
	void VisitOut(const HorseIR::Function *function) override;

	// Statements

	bool VisitIn(const HorseIR::Statement *statement) override;
	void VisitOut(const HorseIR::Statement *statement) override;

	bool VisitIn(const HorseIR::AssignStatement *assignS) override;
	bool VisitIn(const HorseIR::ExpressionStatement *expressionS) override;

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

	// Expressions

	bool VisitIn(const HorseIR::FunctionLiteral *literal) override;
	bool VisitIn(const HorseIR::Identifier *identifier) override;

	static bool IsCompatible(const Geometry *source, const Geometry *destination);

private:
	const UDDUChainsBuilder& m_useDefChains;
	const GeometryAnalysis& m_geometryAnalysis;

	const HorseIR::Statement *m_currentStatement = nullptr;

	CompatibilityGraph *m_graph = new CompatibilityGraph();
	CompatibilityOverlay *m_graphOverlay = nullptr;
};

}
