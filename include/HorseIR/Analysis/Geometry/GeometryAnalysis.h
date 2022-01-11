#pragma once

#include <string>
#include <vector>

#include "HorseIR/Analysis/Framework/StatementAnalysis.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "HorseIR/Analysis/Shape/ShapeAnalysis.h"

#include "HorseIR/Tree/Tree.h"

#include "Libraries/robin_hood.h"

namespace HorseIR {
namespace Analysis {

class GeometryAnalysis : public ConstHierarchicalVisitor, public StatementAnalysis
{
public:
	inline const static std::string Name = "Geometry analysis";
	inline const static std::string ShortName = "geom";

	// Public API

	GeometryAnalysis(const ShapeAnalysis& shapeAnalysis) : m_shapeAnalysis(shapeAnalysis) {}

	void Analyze(const Function *function);
	const Shape *GetGeometry(const Statement *statement) const { return m_geometries.at(statement); }

	// Visitors

	bool VisitIn(const Statement *statement) override;
	void VisitOut(const Statement *statement) override;

	bool VisitIn(const DeclarationStatement *declarationS) override;

	bool VisitIn(const IfStatement *whileS) override;
	bool VisitIn(const WhileStatement *whileS) override;
	bool VisitIn(const RepeatStatement *whileS) override;

	bool VisitIn(const AssignStatement *assignS) override;

	bool VisitIn(const CallExpression *call) override;
	bool VisitIn(const Operand *operand) override;

	// Formatting

	std::string DebugString(const Statement *statement, unsigned int indent = 0) const override;

private:
	const Shape *AnalyzeCall(const FunctionDeclaration *function, const std::vector<const Shape *>& returnShapes, const std::vector<const Shape *>& argumentShapes, const std::vector<const Operand *>& arguments);
	const Shape *AnalyzeCall(const Function *function, const std::vector<const Shape *>& returnShapes, const std::vector<const Shape *>& argumentShapes, const std::vector<const Operand *>& arguments);
	const Shape *AnalyzeCall(const BuiltinFunction *function, const std::vector<const Shape *>& returnShapes, const std::vector<const Shape *>& argumentShapes, const std::vector<const Operand *>& arguments);

	const ShapeAnalysis& m_shapeAnalysis;

	const Statement *m_currentStatement = nullptr;
	const CallExpression *m_call = nullptr;

	const Shape *m_currentGeometry = nullptr;
	robin_hood::unordered_map<const Statement *, const Shape *> m_geometries;
};

}
}
