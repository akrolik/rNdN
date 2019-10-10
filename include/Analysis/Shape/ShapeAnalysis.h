#pragma once

#include <sstream>

#include "HorseIR/Analysis/ForwardAnalysis.h"

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

using ShapeAnalysisProperties = HorseIR::FlowAnalysisMap<SymbolObject, ShapeAnalysisValue>; 
 
class ShapeAnalysis : public HorseIR::ForwardAnalysis<ShapeAnalysisProperties>
{
public:
	using Properties = ShapeAnalysisProperties;

	using HorseIR::ForwardAnalysis<Properties>::ForwardAnalysis;

	void Visit(const HorseIR::Parameter *parameter) override;
	void Visit(const HorseIR::DeclarationStatement *declarations) override;
	void Visit(const HorseIR::AssignStatement *assignS) override;
	void Visit(const HorseIR::ExpressionStatement *expressionS) override;
	void Visit(const HorseIR::BlockStatement *blockS) override;
	void Visit(const HorseIR::IfStatement *ifS) override;
	void Visit(const HorseIR::WhileStatement *whileS) override;
	void Visit(const HorseIR::RepeatStatement *repeatS) override;
	void Visit(const HorseIR::ReturnStatement *returnS) override;

	Properties InitialFlow() const override;
	Properties Merge(const Properties& s1, const Properties& s2) const override;

	std::string Name() const override { return "Shape analysis"; }

	const std::vector<const Shape *>& GetShapes(const HorseIR::Expression *expression) const { return m_expressionShapes.at(expression); }

	const std::vector<const Shape *>& GetReturnShapes() const { return m_returnShapes; }
	const Shape *GetReturnShape(unsigned int i) const { return m_returnShapes.at(i); }

private:
	void CheckCondition(const ShapeAnalysisProperties& shapes, const HorseIR::Operand *operand);

	std::unordered_map<const HorseIR::Expression *, std::vector<const Shape *>> m_expressionShapes;
	std::vector<const Shape *> m_returnShapes;
};

}
