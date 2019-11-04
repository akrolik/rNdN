#pragma once

#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "HorseIR/Analysis/ForwardAnalysis.h"

#include "Analysis/DataObject/DataObject.h"
#include "Analysis/Utils/SymbolObject.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace Analysis {

struct DataObjectValue : HorseIR::FlowAnalysisValue<DataObject>
{
	using Type = DataObject;
	using HorseIR::FlowAnalysisValue<Type>::Equals;

	static void Print(std::ostream& os, const Type *val)
	{
		os << *val;
	}
};

using DataObjectProperties = HorseIR::FlowAnalysisMap<SymbolObject, DataObjectValue>; 

class DataObjectAnalysis : public HorseIR::ForwardAnalysis<DataObjectProperties>
{
public:
	using Properties = DataObjectProperties;
	using HorseIR::ForwardAnalysis<Properties>::ForwardAnalysis;

	// Parameters

	void Visit(const HorseIR::Parameter *parameter) override;

	// Statements

	void Visit(const HorseIR::AssignStatement *assignS) override;
	void Visit(const HorseIR::BlockStatement *blockS) override;
	void Visit(const HorseIR::ReturnStatement *returnS) override;

	// Expressions

	void Visit(const HorseIR::CastExpression *cast) override;
	void Visit(const HorseIR::CallExpression *call) override;
	void Visit(const HorseIR::Identifier *identifier) override;
	void Visit(const HorseIR::Literal *literal) override;

	// Analysis

	Properties InitialFlow() const override;
	Properties Merge(const Properties& s1, const Properties& s2) const override;

	std::string Name() const override { return "Data object analysis"; }

	// Convenience fetching

	const DataObject *GetDataObject(const HorseIR::Operand *operand) const;
	const std::vector<const DataObject *>& GetDataObjects(const HorseIR::Expression *expression) const;

	const DataObject *GetParameterObject(const HorseIR::Parameter *parameter) const { return m_parameterObjects.at(parameter); }
	const std::vector<const DataObject *>& GetReturnObjects() const { return m_returnObjects; }

	// Interprocedural

	const DataObjectAnalysis& GetAnalysis(const HorseIR::Function *function) const { return *m_interproceduralMap.at(function); }

private:
	// Function call visitors

	std::vector<const DataObject *> AnalyzeCall(const HorseIR::FunctionDeclaration *function, const std::vector<const DataObject *>& argumentObjects);
	std::vector<const DataObject *> AnalyzeCall(const HorseIR::Function *function, const std::vector<const DataObject *>& argumentObjects);
	std::vector<const DataObject *> AnalyzeCall(const HorseIR::BuiltinFunction *function, const std::vector<const DataObject *>& argumentObjects);

	std::unordered_map<const HorseIR::Expression *, std::vector<const DataObject *>> m_expressionObjects;
	std::unordered_map<const HorseIR::Parameter *, const DataObject *> m_parameterObjects;
	std::vector<const DataObject *> m_returnObjects;

	// Interprocedural

	std::unordered_map<const HorseIR::Function *, const DataObjectAnalysis *> m_interproceduralMap;
};

}
