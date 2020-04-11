#pragma once

#include <sstream>
#include <utility>

#include "HorseIR/Analysis/ForwardAnalysis.h"

#include "Analysis/DataObject/DataObjectAnalysis.h"

#include "Analysis/Shape/Shape.h"
#include "Analysis/Utils/SymbolObject.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

struct ShapeAnalysisValue
{
	using Type = Shape;

	struct Equals
	{
		 bool operator()(const Type *val1, const Type *val2) const
		 {
			 return val1->Equivalent(*val2);
		 }
	};

	static void Print(std::ostream& os, const Type *val)
	{
		os << *val;
	}
};

using ShapeAnalysisProperties = HorseIR::FlowAnalysisPair<
	HorseIR::FlowAnalysisMap<SymbolObject, ShapeAnalysisValue>, 
	HorseIR::FlowAnalysisMap<SymbolObject, ShapeAnalysisValue>
>;
 
class ShapeAnalysis : public HorseIR::ForwardAnalysis<ShapeAnalysisProperties>
{
public:
	using Properties = ShapeAnalysisProperties;

	ShapeAnalysis(const DataObjectAnalysis& dataAnalysis, const HorseIR::Program *program, bool enforce = false) :
		HorseIR::ForwardAnalysis<ShapeAnalysisProperties>(program), m_dataAnalysis(dataAnalysis), m_enforce(enforce) {}

	void AddCompressionConstraint(const DataObject *dataObject, const Shape::Size *size);

	// Parameters init

	void Visit(const HorseIR::Parameter *parameter) override;

	// Statements

	void Visit(const HorseIR::DeclarationStatement *declarations) override;
	void Visit(const HorseIR::AssignStatement *assignS) override;
	void Visit(const HorseIR::BlockStatement *blockS) override;
	void Visit(const HorseIR::IfStatement *ifS) override;
	void Visit(const HorseIR::WhileStatement *whileS) override;
	void Visit(const HorseIR::RepeatStatement *repeatS) override;
	void Visit(const HorseIR::ReturnStatement *returnS) override;

	// Expressions

	void Visit(const HorseIR::CallExpression *call) override;
	void Visit(const HorseIR::CastExpression *cast) override;
	void Visit(const HorseIR::Identifier *identifier) override;
	void Visit(const HorseIR::VectorLiteral *literal) override;

	// Analysis
	
	Properties InitialFlow() const override;
	Properties Merge(const Properties& s1, const Properties& s2) const override;

	std::string Name() const override { return "Shape analysis"; }

	// Shape utilities for propagation

	void KillShapes(const HorseIR::SymbolTable *symbolTable, HorseIR::FlowAnalysisMap<SymbolObject, ShapeAnalysisValue>& outMap) const;
	void MergeShapes(HorseIR::FlowAnalysisMap<SymbolObject, ShapeAnalysisValue>& outMap, const HorseIR::FlowAnalysisMap<SymbolObject, ShapeAnalysisValue>& otherMap) const;

	const Shape *GetShape(const HorseIR::Operand *operand) const;
	const Shape *GetWriteShape(const HorseIR::Operand *operand) const;

	void SetShape(const HorseIR::Operand *operand, const Shape *shape);
	void SetWriteShape(const HorseIR::Operand *operand, const Shape *shape);

	const std::vector<const Shape *>& GetShapes(const HorseIR::Expression *expression) const;
	const std::vector<const Shape *>& GetWriteShapes(const HorseIR::Expression *expression) const;

	void SetShapes(const HorseIR::Expression *expression, const std::vector<const Shape *>& shapes);
	void SetWriteShapes(const HorseIR::Expression *expression, const std::vector<const Shape *>& shapes);

	// Input and output shapes

	const Shape *GetParameterShape(const HorseIR::Parameter *parameter) const { return m_parameterShapes.at(parameter); }

	const std::vector<const Shape *>& GetReturnShapes() const { return m_returnShapes; }
	const std::vector<const Shape *>& GetReturnWriteShapes() const { return m_returnWriteShapes; }

	const Shape *GetReturnShape(unsigned int i) const { return m_returnShapes.at(i); }
	const Shape *GetReturnWriteShape(unsigned int i) const { return m_returnWriteShapes.at(i); }

	// Interprocedural

	const ShapeAnalysis& GetAnalysis(const HorseIR::Function *function) const { return *m_interproceduralMap.at(function); }
	const DataObjectAnalysis& GetDataAnalysis() const { return m_dataAnalysis; }

private:
	// Function call visitors

	std::pair<std::vector<const Shape *>, std::vector<const Shape *>> AnalyzeCall(const HorseIR::FunctionDeclaration *function, const std::vector<const Shape *>& argumentShapes, const std::vector<HorseIR::Operand *>& arguments);
	std::pair<std::vector<const Shape *>, std::vector<const Shape *>> AnalyzeCall(const HorseIR::Function *function, const std::vector<const Shape *>& argumentShapes, const std::vector<HorseIR::Operand *>& arguments);
	std::pair<std::vector<const Shape *>, std::vector<const Shape *>> AnalyzeCall(const HorseIR::BuiltinFunction *function, const std::vector<const Shape *>& argumentShapes, const std::vector<HorseIR::Operand *>& arguments);

	bool AnalyzeJoinArguments(const std::vector<const Shape *>& argumentShapes, const std::vector<HorseIR::Operand *>& arguments);

	// Static checks for sizes

	bool CheckStaticScalar(const Shape::Size *size) const;
	bool CheckStaticEquality(const Shape::Size *size1, const Shape::Size *size2) const;
	bool CheckConstrainedEquality(const Shape::Size *size1, const Shape::Size *size2) const;
	bool CheckStaticTabular(const ListShape *listShape) const;

	void CheckCondition(const HorseIR::Operand *operand) const;

	std::unordered_map<const DataObject *, const Shape::Size *> m_compressionConstraints;
	bool m_enforce = false;

	// Checks for values

	template<class T>
	std::pair<bool, T> GetConstantArgument(const std::vector<HorseIR::Operand *>& arguments, unsigned int index) const;

	// Utility error function

	[[noreturn]] void ShapeError(const HorseIR::FunctionDeclaration *function, const std::vector<const Shape *>& argumentShapes) const;

	// Shape storage
	
	const HorseIR::CallExpression *m_call = nullptr;

	std::unordered_map<const HorseIR::Expression *, std::vector<const Shape *>> m_expressionShapes;
	std::unordered_map<const HorseIR::Expression *, std::vector<const Shape *>> m_writeShapes;

	std::unordered_map<const HorseIR::Parameter *, const Shape *> m_parameterShapes;

	std::vector<const Shape *> m_returnShapes;
	std::vector<const Shape *> m_returnWriteShapes;

	// Data objects for compression

	const DataObjectAnalysis& m_dataAnalysis;

	// Interprocedural

	std::unordered_map<const HorseIR::Function *, const ShapeAnalysis *> m_interproceduralMap;
};
}
