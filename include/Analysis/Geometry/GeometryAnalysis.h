#pragma once

#include <string>
#include <vector>

#include "HorseIR/Analysis/StatementAnalysis.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "Analysis/Shape/ShapeAnalysis.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

class GeometryAnalysis : public HorseIR::ConstHierarchicalVisitor, public HorseIR::StatementAnalysis
{
public:
	GeometryAnalysis(const ShapeAnalysis& shapeAnalysis) : m_shapeAnalysis(shapeAnalysis) {}

	void Analyze(const HorseIR::Function *function);
	const Shape *GetGeometry(const HorseIR::Statement *statement) const { return m_geometries.at(statement); }

	bool VisitIn(const HorseIR::Statement *statement) override;
	void VisitOut(const HorseIR::Statement *statement) override;

	bool VisitIn(const HorseIR::DeclarationStatement *declarationS) override;

	bool VisitIn(const HorseIR::IfStatement *whileS) override;
	bool VisitIn(const HorseIR::WhileStatement *whileS) override;
	bool VisitIn(const HorseIR::RepeatStatement *whileS) override;

	bool VisitIn(const HorseIR::AssignStatement *assignS) override;

	bool VisitIn(const HorseIR::CallExpression *call) override;
	bool VisitIn(const HorseIR::Operand *operand) override;

	std::string DebugString(const HorseIR::Statement *statement, unsigned int indent = 0) const override;

private:
	const Shape *AnalyzeCall(const HorseIR::FunctionDeclaration *function, const std::vector<HorseIR::Operand *>& arguments);
	const Shape *AnalyzeCall(const HorseIR::Function *function, const std::vector<HorseIR::Operand *>& arguments);
	const Shape *AnalyzeCall(const HorseIR::BuiltinFunction *function, const std::vector<HorseIR::Operand *>& arguments);

	const ShapeAnalysis& m_shapeAnalysis;

	const HorseIR::Statement *m_currentStatement = nullptr;
	const HorseIR::CallExpression *m_call = nullptr;

	const Shape *m_currentGeometry = nullptr;
	std::unordered_map<const HorseIR::Statement *, const Shape *> m_geometries;
};

}
