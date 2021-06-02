#pragma once

#include <stack>
#include <vector>

#include "HorseIR/Traversal/ConstVisitor.h"

#include "HorseIR/Tree/Tree.h"

#include "Libraries/robin_hood.h"

namespace HorseIR {
namespace Transformation {

class OutlineLibrary : public ConstVisitor
{
public:
	OutlineLibrary(std::vector<Function *>& functions, std::stack<robin_hood::unordered_set<const SymbolTable::Symbol *>>& symbols) : m_functions(functions), m_symbols(symbols) {}
	
	Statement *Outline(const Statement *statement);

	// Statements

	void Visit(const Statement *statement) override;
	void Visit(const AssignStatement *assignS) override;
	void Visit(const ExpressionStatement *expressionS) override;

	// Expressions

	void Visit(const Expression *expression) override;
	void Visit(const CallExpression *call) override;

private:
	CallExpression *Outline(const BuiltinFunction *function, const std::vector<const Operand *>& arguments, bool nested = false);

	Function *GenerateInitFunction(const Type *dataType, const BooleanLiteral *orders, bool nested = false);
	Function *GenerateSortFunction(const Type *dataType, const BooleanLiteral *orders, bool shared, bool nested = false);
	Function *GenerateGroupFunction(const Type *dataType);

	Function *GenerateUniqueFunction(const Type *dataType, bool nested = false);

	Function *GenerateJoinHashFunction(const std::vector<const Operand *>& functions, const Type *dataType);
	Function *GenerateJoinCountFunction(const std::vector<const Operand *>& functions, const Type *leftType, const Type *rightType, bool isHashing);
	Function *GenerateJoinFunction(const std::vector<const Operand *>& functions, const Type *leftType, const Type *rightType, bool isHashing);

	Function *GenerateMemberHashFunction(const Type *dataType);
	Function *GenerateMemberFunction(const Type *leftType, const Type *rightType);

	CallExpression *m_libraryCall = nullptr;
	Statement *m_libraryStatement = nullptr;

	std::vector<Function *>& m_functions;
	std::stack<robin_hood::unordered_set<const SymbolTable::Symbol *>>& m_symbols;

	unsigned int m_index = 1u;
};

}
}
