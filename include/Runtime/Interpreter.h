#pragma once

#include <vector>

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Tree/Tree.h"

#include "Runtime/ExecutionEnvironment.h"
#include "Runtime/Runtime.h"
#include "Runtime/DataBuffers/DataBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

namespace Runtime {

class Interpreter : public HorseIR::ConstVisitor
{
public:
	Interpreter(Runtime& runtime) : m_runtime(runtime) {}

	// Function execution engine

	std::vector<DataBuffer *> Execute(const HorseIR::Program *program);

	std::vector<DataBuffer *> Execute(const HorseIR::FunctionDeclaration *function, const std::vector<DataBuffer *>& arguments);
	std::vector<DataBuffer *> Execute(const HorseIR::Function *function, const std::vector<DataBuffer *>& arguments);
	std::vector<DataBuffer *> Execute(const HorseIR::BuiltinFunction *function, const std::vector<DataBuffer *>& arguments);

	// Modules

	void InitializeModule(const HorseIR::Module *module);
	void Visit(const HorseIR::GlobalDeclaration *global) override;

	// Statements

	void Visit(const HorseIR::Statement *statement) override;
	void Visit(const HorseIR::DeclarationStatement *declarationS) override;
	void Visit(const HorseIR::AssignStatement *assignS) override;
	void Visit(const HorseIR::ExpressionStatement *expressionS) override;
	void Visit(const HorseIR::IfStatement *ifS) override;
	void Visit(const HorseIR::WhileStatement *whileS) override;
	void Visit(const HorseIR::RepeatStatement *repeatS) override;
	void Visit(const HorseIR::BlockStatement *blockS) override;
	void Visit(const HorseIR::ReturnStatement *returnS) override;
	void Visit(const HorseIR::BreakStatement *breakS) override;
	void Visit(const HorseIR::ContinueStatement *continueS) override;

	// Expressions

	void Visit(const HorseIR::Expression *expression) override;
	void Visit(const HorseIR::CastExpression *cast) override;
	void Visit(const HorseIR::CallExpression *call) override;
	void Visit(const HorseIR::Identifier *identifier) override;
	void Visit(const HorseIR::FunctionLiteral *literal) override;

	template<typename T>
	void VisitVectorLiteral(const HorseIR::TypedVectorLiteral<T> *literal);

	void Visit(const HorseIR::BooleanLiteral *literal) override;
	void Visit(const HorseIR::CharLiteral *literal) override;
	void Visit(const HorseIR::Int8Literal *literal) override;
	void Visit(const HorseIR::Int16Literal *literal) override;
	void Visit(const HorseIR::Int32Literal *literal) override;
	void Visit(const HorseIR::Int64Literal *literal) override;
	void Visit(const HorseIR::Float32Literal *literal) override;
	void Visit(const HorseIR::Float64Literal *literal) override;
	void Visit(const HorseIR::ComplexLiteral *literal) override;
	void Visit(const HorseIR::StringLiteral *literal) override;
	void Visit(const HorseIR::SymbolLiteral *literal) override;
	void Visit(const HorseIR::DatetimeLiteral *literal) override;
	void Visit(const HorseIR::MonthLiteral *literal) override;
	void Visit(const HorseIR::DateLiteral *literal) override;
	void Visit(const HorseIR::MinuteLiteral *literal) override;
	void Visit(const HorseIR::SecondLiteral *literal) override;
	void Visit(const HorseIR::TimeLiteral *literal) override;

private:
	VectorBuffer *Cast(const HorseIR::BasicType *castType, VectorBuffer *input) const;
	template<class S>
	VectorBuffer *Cast(const HorseIR::BasicType *castType, VectorBuffer *input) const;
	template<class D, class S>
	TypedVectorBuffer<D> *Cast(const HorseIR::BasicType *castType, TypedVectorBuffer<S> *input) const;

	Runtime& m_runtime;
	ExecutionEnvironment m_environment;

	const HorseIR::Program *m_program = nullptr;
};

}
