#pragma once

#include <vector>

#include "HorseIR/Traversal/HierarchicalVisitor.h"

#include "HorseIR/Tree/Expressions/Operand.h"
#include "HorseIR/Tree/Types/Type.h"
#include "HorseIR/Tree/Types/FunctionType.h"

namespace HorseIR {

class TypeChecker : public HierarchicalVisitor
{
public:
	using HierarchicalVisitor::VisitIn;
	using HierarchicalVisitor::VisitOut;

	void Analyze(Program *program);
	
	// Modules

	void VisitOut(GlobalDeclaration *global) override;

	bool VisitIn(Function *function) override;
	void VisitOut(Function *function) override;

	// Statements

	void VisitOut(AssignStatement *assignS) override;
	void VisitOut(ExpressionStatement *expressionS) override;
	void VisitOut(IfStatement *ifS) override;
	void VisitOut(WhileStatement *whileS) override;
	void VisitOut(RepeatStatement *repeatS) override;
	void VisitOut(ReturnStatement *returnS) override;

	// Expressions

	void VisitOut(CastExpression *cast) override;
	void VisitOut(CallExpression *call) override;
	void VisitOut(Identifier *identifier) override;

	// Literals

	void VisitOut(VectorLiteral *literal) override;
	void VisitOut(FunctionLiteral *literal) override;

	// Types

	void VisitOut(EnumerationType *type) override;

private:
	Function *m_currentFunction = nullptr;

	std::vector<Type *> AnalyzeCall(const FunctionType *function, const std::vector<Type *>& argumentTypes);
	std::vector<Type *> AnalyzeCall(const BuiltinFunction *function, const std::vector<Type *>& argumentTypes);
	std::vector<Type *> AnalyzeCall(const Function *function, const FunctionType *functionType, const std::vector<Type *>& argumentTypes);
};

}
