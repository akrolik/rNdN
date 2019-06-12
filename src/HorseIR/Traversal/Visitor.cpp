#include "HorseIR/Traversal/Visitor.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {

// Node superclass

void Visitor::Visit(Node *node)
{

}

// Modules

void Visitor::Visit(Program *program)
{
	Visit(static_cast<Node*>(program));
}

void Visitor::Visit(Module *module)
{
	Visit(static_cast<Node*>(module));
}

void Visitor::Visit(ModuleContent *moduleContent)
{
	Visit(static_cast<Node*>(moduleContent));
}

void Visitor::Visit(ImportDirective *import)
{
	Visit(static_cast<ModuleContent*>(import));
}

void Visitor::Visit(GlobalDeclaration *global)
{
	Visit(static_cast<ModuleContent*>(global));
}

void Visitor::Visit(FunctionDeclaration *function)
{
	Visit(static_cast<ModuleContent*>(function));
}

void Visitor::Visit(BuiltinFunction *function)
{
	Visit(static_cast<FunctionDeclaration*>(function));
}

void Visitor::Visit(Function *function)
{
	Visit(static_cast<FunctionDeclaration*>(function));
}

void Visitor::Visit(VariableDeclaration *declaration)
{
	Visit(static_cast<Node*>(declaration));
}

void Visitor::Visit(Parameter *parameter)
{
	Visit(static_cast<VariableDeclaration*>(parameter));
}

// Statements

void Visitor::Visit(Statement *statement)
{
	Visit(static_cast<Node*>(statement));
}

void Visitor::Visit(DeclarationStatement *declarationS)
{
	Visit(static_cast<Statement*>(declarationS));
}

void Visitor::Visit(AssignStatement *assignS)
{
	Visit(static_cast<Statement*>(assignS));
}

void Visitor::Visit(ExpressionStatement *expressionS)
{
	Visit(static_cast<Statement*>(expressionS));
}

void Visitor::Visit(IfStatement *ifS)
{
	Visit(static_cast<Statement*>(ifS));
}

void Visitor::Visit(WhileStatement *whileS)
{
	Visit(static_cast<Statement*>(whileS));
}

void Visitor::Visit(RepeatStatement *repeatS)
{
	Visit(static_cast<Statement*>(repeatS));
}

void Visitor::Visit(BlockStatement *blockS)
{
	Visit(static_cast<Statement*>(blockS));
}

void Visitor::Visit(ReturnStatement *returnS)
{
	Visit(static_cast<Statement*>(returnS));
}

void Visitor::Visit(BreakStatement *breakS)
{
	Visit(static_cast<Statement*>(breakS));
}

void Visitor::Visit(ContinueStatement *continueS)
{
	Visit(static_cast<Statement*>(continueS));
}            

// Expressions

void Visitor::Visit(Expression *expression)
{
	Visit(static_cast<Node*>(expression));
}

void Visitor::Visit(CallExpression *call)
{
	Visit(static_cast<Expression*>(call));
}

void Visitor::Visit(CastExpression *cast)
{
	Visit(static_cast<Expression*>(cast));
}

void Visitor::Visit(Operand *operand)
{
	Visit(static_cast<Expression*>(operand));
}

void Visitor::Visit(Identifier *identifier)
{
	Visit(static_cast<Operand*>(identifier));
}

void Visitor::Visit(Literal *literal)
{
	Visit(static_cast<Operand*>(literal));
}

// Literals
                            
void Visitor::Visit(VectorLiteral *literal)
{
	Visit(static_cast<Literal*>(literal));
}

void Visitor::Visit(BooleanLiteral *literal)
{
	Visit(static_cast<VectorLiteral*>(literal));
}

void Visitor::Visit(CharLiteral *literal)
{
	Visit(static_cast<VectorLiteral*>(literal));
}

void Visitor::Visit(Int8Literal *literal)
{
	Visit(static_cast<VectorLiteral*>(literal));
}

void Visitor::Visit(Int16Literal *literal)
{
	Visit(static_cast<VectorLiteral*>(literal));
}

void Visitor::Visit(Int32Literal *literal)
{
	Visit(static_cast<VectorLiteral*>(literal));
}

void Visitor::Visit(Int64Literal *literal)
{
	Visit(static_cast<VectorLiteral*>(literal));
}

void Visitor::Visit(Float32Literal *literal)
{
	Visit(static_cast<VectorLiteral*>(literal));
}

void Visitor::Visit(Float64Literal *literal)
{
	Visit(static_cast<VectorLiteral*>(literal));
}

void Visitor::Visit(ComplexLiteral *literal)
{
	Visit(static_cast<VectorLiteral*>(literal));
}

void Visitor::Visit(StringLiteral *literal)
{
	Visit(static_cast<VectorLiteral*>(literal));
}

void Visitor::Visit(SymbolLiteral *literal)
{
	Visit(static_cast<VectorLiteral*>(literal));
}

void Visitor::Visit(DatetimeLiteral *literal)
{
	Visit(static_cast<VectorLiteral*>(literal));
}

void Visitor::Visit(MonthLiteral *literal)
{
	Visit(static_cast<VectorLiteral*>(literal));
}

void Visitor::Visit(DateLiteral *literal)
{
	Visit(static_cast<VectorLiteral*>(literal));
}

void Visitor::Visit(MinuteLiteral *literal)
{
	Visit(static_cast<VectorLiteral*>(literal));
}

void Visitor::Visit(SecondLiteral *literal)
{
	Visit(static_cast<VectorLiteral*>(literal));
}

void Visitor::Visit(TimeLiteral *literal)
{
	Visit(static_cast<VectorLiteral*>(literal));
}

void Visitor::Visit(FunctionLiteral *literal)
{
	Visit(static_cast<Literal*>(literal));
}

// Types

void Visitor::Visit(Type *type)
{
	Visit(static_cast<Node*>(type));
}

void Visitor::Visit(WildcardType *type)
{
	Visit(static_cast<Type*>(type));
}

void Visitor::Visit(BasicType *type)
{
	Visit(static_cast<Type*>(type));
}

void Visitor::Visit(FunctionType *type)
{
	Visit(static_cast<Type*>(type));
}

void Visitor::Visit(ListType *type)
{
	Visit(static_cast<Type*>(type));
}

void Visitor::Visit(DictionaryType *type)
{
	Visit(static_cast<Type*>(type));
}

void Visitor::Visit(EnumerationType *type)
{
	Visit(static_cast<Type*>(type));
}

void Visitor::Visit(TableType *type)
{
	Visit(static_cast<Type*>(type));
}

void Visitor::Visit(KeyedTableType *type)
{
	Visit(static_cast<Type*>(type));
}

}
