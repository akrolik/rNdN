#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {

// Node superclass

bool ConstHierarchicalVisitor::VisitIn(const Node *node)
{
	return true;
}

void ConstHierarchicalVisitor::VisitOut(const Node *node)
{

}

// Modules

bool ConstHierarchicalVisitor::VisitIn(const Program *program)
{
	return VisitIn(static_cast<const Node*>(program));
}

bool ConstHierarchicalVisitor::VisitIn(const Module *module)
{
	return VisitIn(static_cast<const Node*>(module));
}

bool ConstHierarchicalVisitor::VisitIn(const LibraryModule *module)
{
	return VisitIn(static_cast<const Module*>(module));
}

bool ConstHierarchicalVisitor::VisitIn(const ModuleContent *moduleContent)
{
	return VisitIn(static_cast<const Node*>(moduleContent));
}

bool ConstHierarchicalVisitor::VisitIn(const ImportDirective *import)
{
	return VisitIn(static_cast<const ModuleContent*>(import));
}

bool ConstHierarchicalVisitor::VisitIn(const GlobalDeclaration *global)
{
	return VisitIn(static_cast<const ModuleContent*>(global));
}

bool ConstHierarchicalVisitor::VisitIn(const FunctionDeclaration *function)
{
	return VisitIn(static_cast<const ModuleContent*>(function));
}

bool ConstHierarchicalVisitor::VisitIn(const BuiltinFunction *function)
{
	return VisitIn(static_cast<const FunctionDeclaration*>(function));
}

bool ConstHierarchicalVisitor::VisitIn(const Function *function)
{
	return VisitIn(static_cast<const FunctionDeclaration*>(function));
}

bool ConstHierarchicalVisitor::VisitIn(const VariableDeclaration *declaration)
{
	return VisitIn(static_cast<const Node*>(declaration));
}

bool ConstHierarchicalVisitor::VisitIn(const Parameter *parameter)
{
	return VisitIn(static_cast<const VariableDeclaration*>(parameter));
}

void ConstHierarchicalVisitor::VisitOut(const Program *program)
{
	VisitOut(static_cast<const Node*>(program));
}

void ConstHierarchicalVisitor::VisitOut(const Module *module)
{
	VisitOut(static_cast<const Node*>(module));
}

void ConstHierarchicalVisitor::VisitOut(const LibraryModule *module)
{
	VisitOut(static_cast<const Module*>(module));
}

void ConstHierarchicalVisitor::VisitOut(const ModuleContent *moduleContent)
{
	VisitOut(static_cast<const Node*>(moduleContent));
}

void ConstHierarchicalVisitor::VisitOut(const ImportDirective *import)
{
	VisitOut(static_cast<const ModuleContent*>(import));
}

void ConstHierarchicalVisitor::VisitOut(const GlobalDeclaration *global)
{
	VisitOut(static_cast<const ModuleContent*>(global));
}

void ConstHierarchicalVisitor::VisitOut(const FunctionDeclaration *function)
{
	VisitOut(static_cast<const ModuleContent*>(function));
}

void ConstHierarchicalVisitor::VisitOut(const BuiltinFunction *function)
{
	VisitOut(static_cast<const FunctionDeclaration*>(function));
}

void ConstHierarchicalVisitor::VisitOut(const Function *function)
{
	VisitOut(static_cast<const FunctionDeclaration*>(function));
}

void ConstHierarchicalVisitor::VisitOut(const VariableDeclaration *declaration)
{
	VisitOut(static_cast<const Node*>(declaration));
}

void ConstHierarchicalVisitor::VisitOut(const Parameter *parameter)
{
	VisitOut(static_cast<const VariableDeclaration*>(parameter));
}

// Statements

bool ConstHierarchicalVisitor::VisitIn(const Statement *statement)
{
	return VisitIn(static_cast<const Node*>(statement));
}

bool ConstHierarchicalVisitor::VisitIn(const DeclarationStatement *declarationS)
{
	return VisitIn(static_cast<const Statement*>(declarationS));
}

bool ConstHierarchicalVisitor::VisitIn(const AssignStatement *assignS)
{
	return VisitIn(static_cast<const Statement*>(assignS));
}

bool ConstHierarchicalVisitor::VisitIn(const ExpressionStatement *expressionS)
{
	return VisitIn(static_cast<const Statement*>(expressionS));
}

bool ConstHierarchicalVisitor::VisitIn(const IfStatement *ifS)
{
	return VisitIn(static_cast<const Statement*>(ifS));
}

bool ConstHierarchicalVisitor::VisitIn(const WhileStatement *whileS)
{
	return VisitIn(static_cast<const Statement*>(whileS));
}

bool ConstHierarchicalVisitor::VisitIn(const RepeatStatement *repeatS)
{
	return VisitIn(static_cast<const Statement*>(repeatS));
}

bool ConstHierarchicalVisitor::VisitIn(const BlockStatement *blockS)
{
	return VisitIn(static_cast<const Statement*>(blockS));
}

bool ConstHierarchicalVisitor::VisitIn(const ReturnStatement *returnS)
{
	return VisitIn(static_cast<const Statement*>(returnS));
}

bool ConstHierarchicalVisitor::VisitIn(const BreakStatement *breakS)
{
	return VisitIn(static_cast<const Statement*>(breakS));
}

bool ConstHierarchicalVisitor::VisitIn(const ContinueStatement *continueS)
{
	return VisitIn(static_cast<const Statement*>(continueS));
}            

void ConstHierarchicalVisitor::VisitOut(const Statement *statement)
{
	VisitOut(static_cast<const Node*>(statement));
}

void ConstHierarchicalVisitor::VisitOut(const DeclarationStatement *declarationS)
{
	VisitOut(static_cast<const Statement*>(declarationS));
}

void ConstHierarchicalVisitor::VisitOut(const AssignStatement *assignS)
{
	VisitOut(static_cast<const Statement*>(assignS));
}

void ConstHierarchicalVisitor::VisitOut(const ExpressionStatement *expressionS)
{
	VisitOut(static_cast<const Statement*>(expressionS));
}

void ConstHierarchicalVisitor::VisitOut(const IfStatement *ifS)
{
	VisitOut(static_cast<const Statement*>(ifS));
}

void ConstHierarchicalVisitor::VisitOut(const WhileStatement *whileS)
{
	VisitOut(static_cast<const Statement*>(whileS));
}

void ConstHierarchicalVisitor::VisitOut(const RepeatStatement *repeatS)
{
	VisitOut(static_cast<const Statement*>(repeatS));
}

void ConstHierarchicalVisitor::VisitOut(const BlockStatement *blockS)
{
	VisitOut(static_cast<const Statement*>(blockS));
}

void ConstHierarchicalVisitor::VisitOut(const ReturnStatement *returnS)
{
	VisitOut(static_cast<const Statement*>(returnS));
}

void ConstHierarchicalVisitor::VisitOut(const BreakStatement *breakS)
{
	VisitOut(static_cast<const Statement*>(breakS));
}            

void ConstHierarchicalVisitor::VisitOut(const ContinueStatement *continueS)
{
	VisitOut(static_cast<const Statement*>(continueS));
}            

// Expressions

bool ConstHierarchicalVisitor::VisitIn(const Expression *expression)
{
	return VisitIn(static_cast<const Node*>(expression));
}

bool ConstHierarchicalVisitor::VisitIn(const CallExpression *call)
{
	return VisitIn(static_cast<const Expression*>(call));
}

bool ConstHierarchicalVisitor::VisitIn(const CastExpression *cast)
{
	return VisitIn(static_cast<const Expression*>(cast));
}

bool ConstHierarchicalVisitor::VisitIn(const Operand *operand)
{
	return VisitIn(static_cast<const Expression*>(operand));
}

bool ConstHierarchicalVisitor::VisitIn(const Identifier *identifier)
{
	return VisitIn(static_cast<const Operand*>(identifier));
}

bool ConstHierarchicalVisitor::VisitIn(const Literal *literal)
{
	return VisitIn(static_cast<const Operand*>(literal));
}

void ConstHierarchicalVisitor::VisitOut(const Expression *expression)
{
	VisitOut(static_cast<const Node*>(expression));
}

void ConstHierarchicalVisitor::VisitOut(const CallExpression *call)
{
	VisitOut(static_cast<const Expression*>(call));
}

void ConstHierarchicalVisitor::VisitOut(const CastExpression *cast)
{
	VisitOut(static_cast<const Expression*>(cast));
}

void ConstHierarchicalVisitor::VisitOut(const Operand *operand)
{
	VisitOut(static_cast<const Expression*>(operand));
}

void ConstHierarchicalVisitor::VisitOut(const Identifier *identifier)
{
	VisitOut(static_cast<const Operand*>(identifier));
}

void ConstHierarchicalVisitor::VisitOut(const Literal *literal)
{
	VisitOut(static_cast<const Operand*>(literal));
}

// Literals
                            
bool ConstHierarchicalVisitor::VisitIn(const VectorLiteral *literal)
{
	return VisitIn(static_cast<const Literal*>(literal));
}

bool ConstHierarchicalVisitor::VisitIn(const BooleanLiteral *literal)
{
	return VisitIn(static_cast<const VectorLiteral*>(literal));
}

bool ConstHierarchicalVisitor::VisitIn(const CharLiteral *literal)
{
	return VisitIn(static_cast<const VectorLiteral*>(literal));
}

bool ConstHierarchicalVisitor::VisitIn(const Int8Literal *literal)
{
	return VisitIn(static_cast<const VectorLiteral*>(literal));
}

bool ConstHierarchicalVisitor::VisitIn(const Int16Literal *literal)
{
	return VisitIn(static_cast<const VectorLiteral*>(literal));
}

bool ConstHierarchicalVisitor::VisitIn(const Int32Literal *literal)
{
	return VisitIn(static_cast<const VectorLiteral*>(literal));
}

bool ConstHierarchicalVisitor::VisitIn(const Int64Literal *literal)
{
	return VisitIn(static_cast<const VectorLiteral*>(literal));
}

bool ConstHierarchicalVisitor::VisitIn(const Float32Literal *literal)
{
	return VisitIn(static_cast<const VectorLiteral*>(literal));
}

bool ConstHierarchicalVisitor::VisitIn(const Float64Literal *literal)
{
	return VisitIn(static_cast<const VectorLiteral*>(literal));
}

bool ConstHierarchicalVisitor::VisitIn(const ComplexLiteral *literal)
{
	return VisitIn(static_cast<const VectorLiteral*>(literal));
}

bool ConstHierarchicalVisitor::VisitIn(const StringLiteral *literal)
{
	return VisitIn(static_cast<const VectorLiteral*>(literal));
}

bool ConstHierarchicalVisitor::VisitIn(const SymbolLiteral *literal)
{
	return VisitIn(static_cast<const VectorLiteral*>(literal));
}

bool ConstHierarchicalVisitor::VisitIn(const DatetimeLiteral *literal)
{
	return VisitIn(static_cast<const VectorLiteral*>(literal));
}

bool ConstHierarchicalVisitor::VisitIn(const MonthLiteral *literal)
{
	return VisitIn(static_cast<const VectorLiteral*>(literal));
}

bool ConstHierarchicalVisitor::VisitIn(const DateLiteral *literal)
{
	return VisitIn(static_cast<const VectorLiteral*>(literal));
}

bool ConstHierarchicalVisitor::VisitIn(const MinuteLiteral *literal)
{
	return VisitIn(static_cast<const VectorLiteral*>(literal));
}

bool ConstHierarchicalVisitor::VisitIn(const SecondLiteral *literal)
{
	return VisitIn(static_cast<const VectorLiteral*>(literal));
}

bool ConstHierarchicalVisitor::VisitIn(const TimeLiteral *literal)
{
	return VisitIn(static_cast<const VectorLiteral*>(literal));
}

bool ConstHierarchicalVisitor::VisitIn(const FunctionLiteral *literal)
{
	return VisitIn(static_cast<const Literal*>(literal));
}

void ConstHierarchicalVisitor::VisitOut(const VectorLiteral *literal)
{
	VisitOut(static_cast<const Literal*>(literal));
}

void ConstHierarchicalVisitor::VisitOut(const BooleanLiteral *literal)
{
	VisitOut(static_cast<const VectorLiteral*>(literal));
}

void ConstHierarchicalVisitor::VisitOut(const CharLiteral *literal)
{
	VisitOut(static_cast<const VectorLiteral*>(literal));
}

void ConstHierarchicalVisitor::VisitOut(const Int8Literal *literal)
{
	VisitOut(static_cast<const VectorLiteral*>(literal));
}

void ConstHierarchicalVisitor::VisitOut(const Int16Literal *literal)
{
	VisitOut(static_cast<const VectorLiteral*>(literal));
}

void ConstHierarchicalVisitor::VisitOut(const Int32Literal *literal)
{
	VisitOut(static_cast<const VectorLiteral*>(literal));
}

void ConstHierarchicalVisitor::VisitOut(const Int64Literal *literal)
{
	VisitOut(static_cast<const VectorLiteral*>(literal));
}

void ConstHierarchicalVisitor::VisitOut(const Float32Literal *literal)
{
	VisitOut(static_cast<const VectorLiteral*>(literal));
}

void ConstHierarchicalVisitor::VisitOut(const Float64Literal *literal)
{
	VisitOut(static_cast<const VectorLiteral*>(literal));
}

void ConstHierarchicalVisitor::VisitOut(const ComplexLiteral *literal)
{
	VisitOut(static_cast<const VectorLiteral*>(literal));
}

void ConstHierarchicalVisitor::VisitOut(const StringLiteral *literal)
{
	VisitOut(static_cast<const VectorLiteral*>(literal));
}

void ConstHierarchicalVisitor::VisitOut(const SymbolLiteral *literal)
{
	VisitOut(static_cast<const VectorLiteral*>(literal));
}

void ConstHierarchicalVisitor::VisitOut(const DatetimeLiteral *literal)
{
	VisitOut(static_cast<const VectorLiteral*>(literal));
}

void ConstHierarchicalVisitor::VisitOut(const MonthLiteral *literal)
{
	VisitOut(static_cast<const VectorLiteral*>(literal));
}

void ConstHierarchicalVisitor::VisitOut(const DateLiteral *literal)
{
	VisitOut(static_cast<const VectorLiteral*>(literal));
}

void ConstHierarchicalVisitor::VisitOut(const MinuteLiteral *literal)
{
	VisitOut(static_cast<const VectorLiteral*>(literal));
}

void ConstHierarchicalVisitor::VisitOut(const SecondLiteral *literal)
{
	VisitOut(static_cast<const VectorLiteral*>(literal));
}

void ConstHierarchicalVisitor::VisitOut(const TimeLiteral *literal)
{
	VisitOut(static_cast<const VectorLiteral*>(literal));
}

void ConstHierarchicalVisitor::VisitOut(const FunctionLiteral *literal)
{
	VisitOut(static_cast<const Literal*>(literal));
}

// Types

bool ConstHierarchicalVisitor::VisitIn(const Type *type)
{
	return VisitIn(static_cast<const Node*>(type));
}

bool ConstHierarchicalVisitor::VisitIn(const WildcardType *type)
{
	return VisitIn(static_cast<const Type*>(type));
}

bool ConstHierarchicalVisitor::VisitIn(const BasicType *type)
{
	return VisitIn(static_cast<const Type*>(type));
}

bool ConstHierarchicalVisitor::VisitIn(const FunctionType *type)
{
	return VisitIn(static_cast<const Type*>(type));
}

bool ConstHierarchicalVisitor::VisitIn(const ListType *type)
{
	return VisitIn(static_cast<const Type*>(type));
}

bool ConstHierarchicalVisitor::VisitIn(const DictionaryType *type)
{
	return VisitIn(static_cast<const Type*>(type));
}

bool ConstHierarchicalVisitor::VisitIn(const EnumerationType *type)
{
	return VisitIn(static_cast<const Type*>(type));
}

bool ConstHierarchicalVisitor::VisitIn(const TableType *type)
{
	return VisitIn(static_cast<const Type*>(type));
}

bool ConstHierarchicalVisitor::VisitIn(const KeyedTableType *type)
{
	return VisitIn(static_cast<const Type*>(type));
}

void ConstHierarchicalVisitor::VisitOut(const Type *type)
{
	VisitOut(static_cast<const Node*>(type));
}

void ConstHierarchicalVisitor::VisitOut(const WildcardType *type)
{
	VisitOut(static_cast<const Type*>(type));
}

void ConstHierarchicalVisitor::VisitOut(const BasicType *type)
{
	VisitOut(static_cast<const Type*>(type));
}

void ConstHierarchicalVisitor::VisitOut(const FunctionType *type)
{
	VisitOut(static_cast<const Type*>(type));
}

void ConstHierarchicalVisitor::VisitOut(const ListType *type)
{
	VisitOut(static_cast<const Type*>(type));
}

void ConstHierarchicalVisitor::VisitOut(const DictionaryType *type)
{
	VisitOut(static_cast<const Type*>(type));
}

void ConstHierarchicalVisitor::VisitOut(const EnumerationType *type)
{
	VisitOut(static_cast<const Type*>(type));
}

void ConstHierarchicalVisitor::VisitOut(const TableType *type)
{
	VisitOut(static_cast<const Type*>(type));
}

void ConstHierarchicalVisitor::VisitOut(const KeyedTableType *type)
{
	VisitOut(static_cast<const Type*>(type));
}

}
