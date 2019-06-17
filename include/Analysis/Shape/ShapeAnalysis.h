#pragma once

#include "HorseIR/Analysis/ForwardAnalysis.h"

#include "Analysis/Shape/Shape.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {
                                                       
struct ShapeAnalysisKey : HorseIR::FlowAnalysisPointerValue<HorseIR::SymbolTable::Symbol>
{
	using Type = HorseIR::SymbolTable::Symbol;
	using HorseIR::FlowAnalysisPointerValue<Type>::Equals;

	struct Hash
	{
		bool operator()(const Type *val) const
		{
			return std::hash<const Type *>()(val);
		}
	};

	static void Print(std::ostream& os, const Type *val)
	{
		os << HorseIR::PrettyPrinter::PrettyString(val->node);
	}
};

struct ShapeAnalysisValue : HorseIR::FlowAnalysisValue<Shape>
{
	using Type = Shape;
	using HorseIR::FlowAnalysisValue<Type>::Equals;

	static void Print(std::ostream& os, const Type *val)
	{
		os << val;
	}
};

using ShapeAnalysisProperties = HorseIR::FlowAnalysisMap<ShapeAnalysisKey, ShapeAnalysisValue>; 
 
class ShapeAnalysis : public HorseIR::ForwardAnalysis<ShapeAnalysisProperties>
{
public:
	using Properties = ShapeAnalysisProperties;
	using HorseIR::ForwardAnalysis<Properties>::ForwardAnalysis;

	void Visit(const HorseIR::AssignStatement *assign) override;

	Properties Merge(const Properties& s1, const Properties& s2) const override;

private:
	const Shape *MergeShape(const Shape *shape1, const Shape *shape2) const;
	const Shape::Size *MergeSize(const Shape::Size *size1, const Shape::Size *size2) const;
};

}
