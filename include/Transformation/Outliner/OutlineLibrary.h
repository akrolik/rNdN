#pragma once

#include <stack>
#include <unordered_set>
#include <vector>

#include "HorseIR/Traversal/ConstVisitor.h"

#include "HorseIR/Tree/Tree.h"

namespace Transformation {

class OutlineLibrary : public HorseIR::ConstVisitor
{
public:
	OutlineLibrary(std::vector<HorseIR::Function *>& functions, std::stack<std::unordered_set<const HorseIR::SymbolTable::Symbol *>>& symbols) : m_functions(functions), m_symbols(symbols) {}
	
	HorseIR::Statement *Outline(const HorseIR::Statement *statement);

	// Statements

	void Visit(const HorseIR::Statement *statement) override;
	void Visit(const HorseIR::AssignStatement *assignS) override;
	void Visit(const HorseIR::ExpressionStatement *expressionS) override;

	// Expressions

	void Visit(const HorseIR::Expression *expression) override;
	void Visit(const HorseIR::CallExpression *call) override;

private:
	HorseIR::CallExpression *Outline(const HorseIR::BuiltinFunction *function, const std::vector<HorseIR::Operand *>& arguments);

	HorseIR::Function *GenerateInitFunction(const HorseIR::Type *dataType, const HorseIR::BooleanLiteral *orders);
	HorseIR::Function *GenerateSortFunction(const HorseIR::Type *dataType, const HorseIR::BooleanLiteral *orders);
	HorseIR::Function *GenerateGroupFunction(const HorseIR::Type *dataType);

	HorseIR::CallExpression *m_libraryCall = nullptr;
	HorseIR::Statement *m_libraryStatement = nullptr;

	std::vector<HorseIR::Function *>& m_functions;
	std::stack<std::unordered_set<const HorseIR::SymbolTable::Symbol *>>& m_symbols;

	unsigned int m_index = 1u;
};

}