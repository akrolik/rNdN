#pragma once

#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "HorseIR/Analysis/Framework/ForwardAnalysis.h"

#include "HorseIR/Analysis/DataObject/DataObject.h"
#include "HorseIR/Analysis/Utils/SymbolObject.h"
#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace HorseIR {
namespace Analysis {

struct DataObjectValue : FlowAnalysisValue<DataObject>
{
	using Type = DataObject;
	using FlowAnalysisValue<Type>::Equals;

	static void Print(std::ostream& os, const Type *val)
	{
		os << *val;
	}
};

using DataObjectProperties = FlowAnalysisMap<SymbolObject, DataObjectValue>; 

class DataObjectAnalysis : public ForwardAnalysis<DataObjectProperties>
{
public:
	using Properties = DataObjectProperties;
	using ForwardAnalysis<Properties>::ForwardAnalysis;

	// Parameters

	void Visit(const Parameter *parameter) override;

	// Statements

	void Visit(const AssignStatement *assignS) override;
	void Visit(const BlockStatement *blockS) override;
	void Visit(const ReturnStatement *returnS) override;

	// Expressions

	void Visit(const CastExpression *cast) override;
	void Visit(const CallExpression *call) override;
	void Visit(const Identifier *identifier) override;
	void Visit(const Literal *literal) override;

	// Analysis

	Properties InitialFlow() const override;
	Properties Merge(const Properties& s1, const Properties& s2) const override;

	std::string Name() const override { return "Data object analysis"; }

	// Convenience fetching

	const DataObject *GetDataObject(const Operand *operand) const;
	const std::vector<const DataObject *>& GetDataObjects(const Expression *expression) const;

	const DataObject *GetParameterObject(const Parameter *parameter) const { return m_parameterObjects.at(parameter); }

	const std::vector<const DataObject *>& GetReturnObjects() const { return m_returnObjects; }
	const DataObject *GetReturnObject(unsigned int index) const { return m_returnObjects.at(index); }

	// Interprocedural

	const DataObjectAnalysis& GetAnalysis(const Function *function) const { return *m_interproceduralMap.at(function); }

private:
	// Function call visitors

	std::vector<const DataObject *> AnalyzeCall(const FunctionDeclaration *function, const std::vector<const Operand *>& arguments, const std::vector<const DataObject *>& argumentObjects);
	std::vector<const DataObject *> AnalyzeCall(const Function *function, const std::vector<const Operand *>& arguments, const std::vector<const DataObject *>& argumentObjects);
	std::vector<const DataObject *> AnalyzeCall(const BuiltinFunction *function, const std::vector<const Operand *>& arguments, const std::vector<const DataObject *>& argumentObjects);

	std::unordered_map<const Expression *, std::vector<const DataObject *>> m_expressionObjects;
	std::unordered_map<const Parameter *, const DataObject *> m_parameterObjects;
	std::vector<const DataObject *> m_returnObjects;

	// Interprocedural

	std::unordered_map<const Function *, const DataObjectAnalysis *> m_interproceduralMap;
};

}
}
