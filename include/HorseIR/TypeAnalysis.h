#pragma once

#include <vector>

#include "HorseIR/Traversal/ForwardTraversal.h"

#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class TypeAnalysis : public ForwardTraversal
{
public:
	void Analyze(Method *method);

	void Visit(AssignStatement *assign) override;

	void Visit(CallExpression *call) override;
	void Visit(CastExpression *cast) override;
	void Visit(Identifier *identifier) override;
	void Visit(Literal<int64_t> *literal) override;
	void Visit(Literal<double> *literal) override;
	void Visit(Literal<std::string> *literal) override;
	void Visit(Symbol *symbol) override;

private:
	const Type *AnalyzeCall(const BuiltinMethod *method, const std::vector<Expression *>& arguments);
};

}
