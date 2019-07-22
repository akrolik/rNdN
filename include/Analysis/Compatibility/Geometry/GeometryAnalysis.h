#pragma once

#include <vector>

#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "Analysis/Compatibility/Geometry/Geometry.h"
#include "Analysis/Shape/ShapeAnalysis.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

class GeometryAnalysis : public HorseIR::ConstHierarchicalVisitor
{
public:
	GeometryAnalysis(const ShapeAnalysis& shapeAnalysis) : m_shapeAnalysis(shapeAnalysis) {}

	void Analyze(const HorseIR::Function *function);
	const Geometry *GetGeometry(const HorseIR::Statement *statement) const { return m_geometries.at(statement); }

	bool VisitIn(const HorseIR::Statement *statement) override;
	void VisitOut(const HorseIR::Statement *statement) override;

	bool VisitIn(const HorseIR::IfStatement *whileS) override;
	bool VisitIn(const HorseIR::WhileStatement *whileS) override;
	bool VisitIn(const HorseIR::RepeatStatement *whileS) override;

	bool VisitIn(const HorseIR::AssignStatement *assignS) override;

	bool VisitIn(const HorseIR::CallExpression *call) override;
	bool VisitIn(const HorseIR::Operand *operand) override;

private:
	Geometry *AnalyzeCall(const HorseIR::FunctionDeclaration *function, const std::vector<HorseIR::Operand *>& arguments);
	Geometry *AnalyzeCall(const HorseIR::Function *function, const std::vector<HorseIR::Operand *>& arguments);
	Geometry *AnalyzeCall(const HorseIR::BuiltinFunction *function, const std::vector<HorseIR::Operand *>& arguments);

	const ShapeAnalysis& m_shapeAnalysis;

	const HorseIR::Statement *m_currentStatement = nullptr;
	const HorseIR::CallExpression *m_call = nullptr;

	const Geometry *m_currentGeometry = nullptr;
	std::unordered_map<const HorseIR::Statement *, const Geometry *> m_geometries;
};

}
