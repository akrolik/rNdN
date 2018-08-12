#pragma once

#include <vector>

#include "HorseIR/Traversal/ForwardTraversal.h"

#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class TypeAnalysis : public ForwardTraversal
{
public:
	void Analyze(Program *program);

	void Visit(Method *method) override;

	void Visit(AssignStatement *assign) override;

	void Visit(CallExpression *call) override;
	void Visit(CastExpression *cast) override;
	void Visit(Identifier *identifier) override;
	void Visit(FunctionLiteral *literal) override;

	void Visit(ReturnStatement *ret) override;

private:
	Method *m_currentMethod = nullptr;

	Type *AnalyzeCall(const MethodDeclaration *method, const std::vector<Type *>& arguments);
	Type *AnalyzeCall(const Method *method, const std::vector<Type *>& arguments);
	Type *AnalyzeCall(const BuiltinMethod *method, const std::vector<Type *>& arguments);
};

}
