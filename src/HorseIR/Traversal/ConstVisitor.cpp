#include "HorseIR/Traversal/ConstVisitor.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {

// Node superclass

void ConstVisitor::Visit(const Node *node)
{

}

void ConstVisitor::Visit(const Program *program)
{
	Visit(static_cast<const Node*>(program));
}

void ConstVisitor::Visit(const Module *module)
{
	Visit(static_cast<const Node*>(module));
}

void ConstVisitor::Visit(const ModuleContent *moduleContent)
{
	Visit(static_cast<const Node*>(moduleContent));
}

void ConstVisitor::Visit(const ImportDirective *import)
{
	Visit(static_cast<const ModuleContent*>(import));
}

void ConstVisitor::Visit(const GlobalDeclaration *global)
{
	Visit(static_cast<const ModuleContent*>(global));
}

void ConstVisitor::Visit(const FunctionDeclaration *function)
{
	Visit(static_cast<const ModuleContent*>(function));
}

void ConstVisitor::Visit(const BuiltinFunction *function)
{
	Visit(static_cast<const FunctionDeclaration*>(function));
}

void ConstVisitor::Visit(const Function *function)
{
	Visit(static_cast<const FunctionDeclaration*>(function));
}

void ConstVisitor::Visit(const VariableDeclaration *declaration)
{
	Visit(static_cast<const Node*>(declaration));
}

void ConstVisitor::Visit(const Parameter *parameter)
{
	Visit(static_cast<const VariableDeclaration*>(parameter));
}

// Statements

void ConstVisitor::Visit(const Statement *statement)
{
	Visit(static_cast<const Node*>(statement));
}

void ConstVisitor::Visit(const DeclarationStatement *declarationS)
{
	Visit(static_cast<const Statement*>(declarationS));
}

void ConstVisitor::Visit(const AssignStatement *assignS)
{
	Visit(static_cast<const Statement*>(assignS));
}

void ConstVisitor::Visit(const ExpressionStatement *expressionS)
{
	Visit(static_cast<const Statement*>(expressionS));
}

void ConstVisitor::Visit(const IfStatement *ifS)
{
	Visit(static_cast<const Statement*>(ifS));
}

void ConstVisitor::Visit(const WhileStatement *whileS)
{
	Visit(static_cast<const Statement*>(whileS));
}

void ConstVisitor::Visit(const RepeatStatement *repeatS)
{
	Visit(static_cast<const Statement*>(repeatS));
}

void ConstVisitor::Visit(const BlockStatement *blockS)
{
	Visit(static_cast<const Statement*>(blockS));
}

void ConstVisitor::Visit(const ReturnStatement *returnS)
{
	Visit(static_cast<const Statement*>(returnS));
}

void ConstVisitor::Visit(const BreakStatement *breakS)
{
	Visit(static_cast<const Statement*>(breakS));
}

void ConstVisitor::Visit(const ContinueStatement *continueS)
{
	Visit(static_cast<const Statement*>(continueS));
}            

// Expressions

void ConstVisitor::Visit(const Expression *expression)
{
	Visit(static_cast<const Node*>(expression));
}

void ConstVisitor::Visit(const CallExpression *call)
{
	Visit(static_cast<const Expression*>(call));
}

void ConstVisitor::Visit(const CastExpression *cast)
{
	Visit(static_cast<const Expression*>(cast));
}

void ConstVisitor::Visit(const Operand *operand)
{
	Visit(static_cast<const Expression*>(operand));
}

void ConstVisitor::Visit(const Identifier *identifier)
{
	Visit(static_cast<const Operand*>(identifier));
}

void ConstVisitor::Visit(const Literal *literal)
{
	Visit(static_cast<const Operand*>(literal));
}

// Literals
                            
void ConstVisitor::Visit(const VectorLiteral *literal)
{
	Visit(static_cast<const Literal*>(literal));
}

void ConstVisitor::Visit(const BooleanLiteral *literal)
{
	Visit(static_cast<const VectorLiteral*>(literal));
}

void ConstVisitor::Visit(const CharLiteral *literal)
{
	Visit(static_cast<const VectorLiteral*>(literal));
}

void ConstVisitor::Visit(const Int8Literal *literal)
{
	Visit(static_cast<const VectorLiteral*>(literal));
}

void ConstVisitor::Visit(const Int16Literal *literal)
{
	Visit(static_cast<const VectorLiteral*>(literal));
}

void ConstVisitor::Visit(const Int32Literal *literal)
{
	Visit(static_cast<const VectorLiteral*>(literal));
}

void ConstVisitor::Visit(const Int64Literal *literal)
{
	Visit(static_cast<const VectorLiteral*>(literal));
}

void ConstVisitor::Visit(const Float32Literal *literal)
{
	Visit(static_cast<const VectorLiteral*>(literal));
}

void ConstVisitor::Visit(const Float64Literal *literal)
{
	Visit(static_cast<const VectorLiteral*>(literal));
}

void ConstVisitor::Visit(const ComplexLiteral *literal)
{
	Visit(static_cast<const VectorLiteral*>(literal));
}

void ConstVisitor::Visit(const StringLiteral *literal)
{
	Visit(static_cast<const VectorLiteral*>(literal));
}

void ConstVisitor::Visit(const SymbolLiteral *literal)
{
	Visit(static_cast<const VectorLiteral*>(literal));
}

void ConstVisitor::Visit(const DatetimeLiteral *literal)
{
	Visit(static_cast<const VectorLiteral*>(literal));
}

void ConstVisitor::Visit(const MonthLiteral *literal)
{
	Visit(static_cast<const VectorLiteral*>(literal));
}

void ConstVisitor::Visit(const DateLiteral *literal)
{
	Visit(static_cast<const VectorLiteral*>(literal));
}

void ConstVisitor::Visit(const MinuteLiteral *literal)
{
	Visit(static_cast<const VectorLiteral*>(literal));
}

void ConstVisitor::Visit(const SecondLiteral *literal)
{
	Visit(static_cast<const VectorLiteral*>(literal));
}

void ConstVisitor::Visit(const TimeLiteral *literal)
{
	Visit(static_cast<const VectorLiteral*>(literal));
}

void ConstVisitor::Visit(const FunctionLiteral *literal)
{
	Visit(static_cast<const Literal*>(literal));
}

// Types

void ConstVisitor::Visit(const Type *type)
{
	Visit(static_cast<const Node*>(type));
}

void ConstVisitor::Visit(const WildcardType *type)
{
	Visit(static_cast<const Type*>(type));
}

void ConstVisitor::Visit(const BasicType *type)
{
	Visit(static_cast<const Type*>(type));
}

void ConstVisitor::Visit(const FunctionType *type)
{
	Visit(static_cast<const Type*>(type));
}

void ConstVisitor::Visit(const ListType *type)
{
	Visit(static_cast<const Type*>(type));
}

void ConstVisitor::Visit(const DictionaryType *type)
{
	Visit(static_cast<const Type*>(type));
}

void ConstVisitor::Visit(const EnumerationType *type)
{
	Visit(static_cast<const Type*>(type));
}

void ConstVisitor::Visit(const TableType *type)
{
	Visit(static_cast<const Type*>(type));
}

void ConstVisitor::Visit(const KeyedTableType *type)
{
	Visit(static_cast<const Type*>(type));
}

}
