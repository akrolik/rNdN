#include "HorseIR/Traversal/HierarchicalVisitor.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {

// Node superclass

bool HierarchicalVisitor::VisitIn(Node *node)
{
	return true;
}

void HierarchicalVisitor::VisitOut(Node *node)
{

}

// Modules

bool HierarchicalVisitor::VisitIn(Program *program)
{
	return VisitIn(static_cast<Node*>(program));
}

bool HierarchicalVisitor::VisitIn(Module *module)
{
	return VisitIn(static_cast<Node*>(module));
}

bool HierarchicalVisitor::VisitIn(ModuleContent *moduleContent)
{
	return VisitIn(static_cast<Node*>(moduleContent));
}

bool HierarchicalVisitor::VisitIn(ImportDirective *import)
{
	return VisitIn(static_cast<ModuleContent*>(import));
}

bool HierarchicalVisitor::VisitIn(GlobalDeclaration *global)
{
	return VisitIn(static_cast<ModuleContent*>(global));
}

bool HierarchicalVisitor::VisitIn(FunctionDeclaration *function)
{
	return VisitIn(static_cast<ModuleContent*>(function));
}

bool HierarchicalVisitor::VisitIn(BuiltinFunction *function)
{
	return VisitIn(static_cast<FunctionDeclaration*>(function));
}

bool HierarchicalVisitor::VisitIn(Function *function)
{
	return VisitIn(static_cast<FunctionDeclaration*>(function));
}

bool HierarchicalVisitor::VisitIn(VariableDeclaration *declaration)
{
	return VisitIn(static_cast<Node*>(declaration));
}

bool HierarchicalVisitor::VisitIn(Parameter *parameter)
{
	return VisitIn(static_cast<VariableDeclaration*>(parameter));
}

void HierarchicalVisitor::VisitOut(Program *program)
{
	VisitOut(static_cast<Node*>(program));
}

void HierarchicalVisitor::VisitOut(Module *module)
{
	VisitOut(static_cast<Node*>(module));
}

void HierarchicalVisitor::VisitOut(ModuleContent *moduleContent)
{
	VisitOut(static_cast<Node*>(moduleContent));
}

void HierarchicalVisitor::VisitOut(ImportDirective *import)
{
	VisitOut(static_cast<ModuleContent*>(import));
}

void HierarchicalVisitor::VisitOut(GlobalDeclaration *global)
{
	VisitOut(static_cast<ModuleContent*>(global));
}

void HierarchicalVisitor::VisitOut(FunctionDeclaration *function)
{
	VisitOut(static_cast<ModuleContent*>(function));
}

void HierarchicalVisitor::VisitOut(BuiltinFunction *function)
{
	VisitOut(static_cast<FunctionDeclaration*>(function));
}

void HierarchicalVisitor::VisitOut(Function *function)
{
	VisitOut(static_cast<FunctionDeclaration*>(function));
}

void HierarchicalVisitor::VisitOut(VariableDeclaration *declaration)
{
	VisitOut(static_cast<Node*>(declaration));
}

void HierarchicalVisitor::VisitOut(Parameter *parameter)
{
	VisitOut(static_cast<VariableDeclaration*>(parameter));
}

// Statements

bool HierarchicalVisitor::VisitIn(Statement *statement)
{
	return VisitIn(static_cast<Node*>(statement));
}

bool HierarchicalVisitor::VisitIn(DeclarationStatement *declarationS)
{
	return VisitIn(static_cast<Statement*>(declarationS));
}

bool HierarchicalVisitor::VisitIn(AssignStatement *assignS)
{
	return VisitIn(static_cast<Statement*>(assignS));
}

bool HierarchicalVisitor::VisitIn(ExpressionStatement *expressionS)
{
	return VisitIn(static_cast<Statement*>(expressionS));
}

bool HierarchicalVisitor::VisitIn(IfStatement *ifS)
{
	return VisitIn(static_cast<Statement*>(ifS));
}

bool HierarchicalVisitor::VisitIn(WhileStatement *whileS)
{
	return VisitIn(static_cast<Statement*>(whileS));
}

bool HierarchicalVisitor::VisitIn(RepeatStatement *repeatS)
{
	return VisitIn(static_cast<Statement*>(repeatS));
}

bool HierarchicalVisitor::VisitIn(BlockStatement *blockS)
{
	return VisitIn(static_cast<Statement*>(blockS));
}

bool HierarchicalVisitor::VisitIn(ReturnStatement *returnS)
{
	return VisitIn(static_cast<Statement*>(returnS));
}

bool HierarchicalVisitor::VisitIn(BreakStatement *breakS)
{
	return VisitIn(static_cast<Statement*>(breakS));
}

bool HierarchicalVisitor::VisitIn(ContinueStatement *continueS)
{
	return VisitIn(static_cast<Statement*>(continueS));
}            

void HierarchicalVisitor::VisitOut(Statement *statement)
{
	VisitOut(static_cast<Node*>(statement));
}

void HierarchicalVisitor::VisitOut(DeclarationStatement *declarationS)
{
	VisitOut(static_cast<Statement*>(declarationS));
}

void HierarchicalVisitor::VisitOut(AssignStatement *assignS)
{
	VisitOut(static_cast<Statement*>(assignS));
}

void HierarchicalVisitor::VisitOut(ExpressionStatement *expressionS)
{
	VisitOut(static_cast<Statement*>(expressionS));
}

void HierarchicalVisitor::VisitOut(IfStatement *ifS)
{
	VisitOut(static_cast<Statement*>(ifS));
}

void HierarchicalVisitor::VisitOut(WhileStatement *whileS)
{
	VisitOut(static_cast<Statement*>(whileS));
}

void HierarchicalVisitor::VisitOut(RepeatStatement *repeatS)
{
	VisitOut(static_cast<Statement*>(repeatS));
}

void HierarchicalVisitor::VisitOut(BlockStatement *blockS)
{
	VisitOut(static_cast<Statement*>(blockS));
}

void HierarchicalVisitor::VisitOut(ReturnStatement *returnS)
{
	VisitOut(static_cast<Statement*>(returnS));
}

void HierarchicalVisitor::VisitOut(BreakStatement *breakS)
{
	VisitOut(static_cast<Statement*>(breakS));
}            

void HierarchicalVisitor::VisitOut(ContinueStatement *continueS)
{
	VisitOut(static_cast<Statement*>(continueS));
}            

// Expressions

bool HierarchicalVisitor::VisitIn(Expression *expression)
{
	return VisitIn(static_cast<Node*>(expression));
}

bool HierarchicalVisitor::VisitIn(CallExpression *call)
{
	return VisitIn(static_cast<Expression*>(call));
}

bool HierarchicalVisitor::VisitIn(CastExpression *cast)
{
	return VisitIn(static_cast<Expression*>(cast));
}

bool HierarchicalVisitor::VisitIn(Operand *operand)
{
	return VisitIn(static_cast<Expression*>(operand));
}

bool HierarchicalVisitor::VisitIn(Identifier *identifier)
{
	return VisitIn(static_cast<Operand*>(identifier));
}

bool HierarchicalVisitor::VisitIn(Literal *literal)
{
	return VisitIn(static_cast<Operand*>(literal));
}

void HierarchicalVisitor::VisitOut(Expression *expression)
{
	VisitOut(static_cast<Node*>(expression));
}

void HierarchicalVisitor::VisitOut(CallExpression *call)
{
	VisitOut(static_cast<Expression*>(call));
}

void HierarchicalVisitor::VisitOut(CastExpression *cast)
{
	VisitOut(static_cast<Expression*>(cast));
}

void HierarchicalVisitor::VisitOut(Operand *operand)
{
	VisitOut(static_cast<Expression*>(operand));
}

void HierarchicalVisitor::VisitOut(Identifier *identifier)
{
	VisitOut(static_cast<Operand*>(identifier));
}

void HierarchicalVisitor::VisitOut(Literal *literal)
{
	VisitOut(static_cast<Operand*>(literal));
}

// Literals
                            
bool HierarchicalVisitor::VisitIn(VectorLiteral *literal)
{
	return VisitIn(static_cast<Literal*>(literal));
}

bool HierarchicalVisitor::VisitIn(BooleanLiteral *literal)
{
	return VisitIn(static_cast<VectorLiteral*>(literal));
}

bool HierarchicalVisitor::VisitIn(CharLiteral *literal)
{
	return VisitIn(static_cast<VectorLiteral*>(literal));
}

bool HierarchicalVisitor::VisitIn(Int8Literal *literal)
{
	return VisitIn(static_cast<VectorLiteral*>(literal));
}

bool HierarchicalVisitor::VisitIn(Int16Literal *literal)
{
	return VisitIn(static_cast<VectorLiteral*>(literal));
}

bool HierarchicalVisitor::VisitIn(Int32Literal *literal)
{
	return VisitIn(static_cast<VectorLiteral*>(literal));
}

bool HierarchicalVisitor::VisitIn(Int64Literal *literal)
{
	return VisitIn(static_cast<VectorLiteral*>(literal));
}

bool HierarchicalVisitor::VisitIn(Float32Literal *literal)
{
	return VisitIn(static_cast<VectorLiteral*>(literal));
}

bool HierarchicalVisitor::VisitIn(Float64Literal *literal)
{
	return VisitIn(static_cast<VectorLiteral*>(literal));
}

bool HierarchicalVisitor::VisitIn(ComplexLiteral *literal)
{
	return VisitIn(static_cast<VectorLiteral*>(literal));
}

bool HierarchicalVisitor::VisitIn(StringLiteral *literal)
{
	return VisitIn(static_cast<VectorLiteral*>(literal));
}

bool HierarchicalVisitor::VisitIn(SymbolLiteral *literal)
{
	return VisitIn(static_cast<VectorLiteral*>(literal));
}

bool HierarchicalVisitor::VisitIn(DatetimeLiteral *literal)
{
	return VisitIn(static_cast<VectorLiteral*>(literal));
}

bool HierarchicalVisitor::VisitIn(MonthLiteral *literal)
{
	return VisitIn(static_cast<VectorLiteral*>(literal));
}

bool HierarchicalVisitor::VisitIn(DateLiteral *literal)
{
	return VisitIn(static_cast<VectorLiteral*>(literal));
}

bool HierarchicalVisitor::VisitIn(MinuteLiteral *literal)
{
	return VisitIn(static_cast<VectorLiteral*>(literal));
}

bool HierarchicalVisitor::VisitIn(SecondLiteral *literal)
{
	return VisitIn(static_cast<VectorLiteral*>(literal));
}

bool HierarchicalVisitor::VisitIn(TimeLiteral *literal)
{
	return VisitIn(static_cast<VectorLiteral*>(literal));
}

bool HierarchicalVisitor::VisitIn(FunctionLiteral *literal)
{
	return VisitIn(static_cast<Literal*>(literal));
}

void HierarchicalVisitor::VisitOut(VectorLiteral *literal)
{
	VisitOut(static_cast<Literal*>(literal));
}

void HierarchicalVisitor::VisitOut(BooleanLiteral *literal)
{
	VisitOut(static_cast<VectorLiteral*>(literal));
}

void HierarchicalVisitor::VisitOut(CharLiteral *literal)
{
	VisitOut(static_cast<VectorLiteral*>(literal));
}

void HierarchicalVisitor::VisitOut(Int8Literal *literal)
{
	VisitOut(static_cast<VectorLiteral*>(literal));
}

void HierarchicalVisitor::VisitOut(Int16Literal *literal)
{
	VisitOut(static_cast<VectorLiteral*>(literal));
}

void HierarchicalVisitor::VisitOut(Int32Literal *literal)
{
	VisitOut(static_cast<VectorLiteral*>(literal));
}

void HierarchicalVisitor::VisitOut(Int64Literal *literal)
{
	VisitOut(static_cast<VectorLiteral*>(literal));
}

void HierarchicalVisitor::VisitOut(Float32Literal *literal)
{
	VisitOut(static_cast<VectorLiteral*>(literal));
}

void HierarchicalVisitor::VisitOut(Float64Literal *literal)
{
	VisitOut(static_cast<VectorLiteral*>(literal));
}

void HierarchicalVisitor::VisitOut(ComplexLiteral *literal)
{
	VisitOut(static_cast<VectorLiteral*>(literal));
}

void HierarchicalVisitor::VisitOut(StringLiteral *literal)
{
	VisitOut(static_cast<VectorLiteral*>(literal));
}

void HierarchicalVisitor::VisitOut(SymbolLiteral *literal)
{
	VisitOut(static_cast<VectorLiteral*>(literal));
}

void HierarchicalVisitor::VisitOut(DatetimeLiteral *literal)
{
	VisitOut(static_cast<VectorLiteral*>(literal));
}

void HierarchicalVisitor::VisitOut(MonthLiteral *literal)
{
	VisitOut(static_cast<VectorLiteral*>(literal));
}

void HierarchicalVisitor::VisitOut(DateLiteral *literal)
{
	VisitOut(static_cast<VectorLiteral*>(literal));
}

void HierarchicalVisitor::VisitOut(MinuteLiteral *literal)
{
	VisitOut(static_cast<VectorLiteral*>(literal));
}

void HierarchicalVisitor::VisitOut(SecondLiteral *literal)
{
	VisitOut(static_cast<VectorLiteral*>(literal));
}

void HierarchicalVisitor::VisitOut(TimeLiteral *literal)
{
	VisitOut(static_cast<VectorLiteral*>(literal));
}

void HierarchicalVisitor::VisitOut(FunctionLiteral *literal)
{
	VisitOut(static_cast<Literal*>(literal));
}

// Types

bool HierarchicalVisitor::VisitIn(Type *type)
{
	return VisitIn(static_cast<Node*>(type));
}

bool HierarchicalVisitor::VisitIn(WildcardType *type)
{
	return VisitIn(static_cast<Type*>(type));
}

bool HierarchicalVisitor::VisitIn(BasicType *type)
{
	return VisitIn(static_cast<Type*>(type));
}

bool HierarchicalVisitor::VisitIn(FunctionType *type)
{
	return VisitIn(static_cast<Type*>(type));
}

bool HierarchicalVisitor::VisitIn(ListType *type)
{
	return VisitIn(static_cast<Type*>(type));
}

bool HierarchicalVisitor::VisitIn(DictionaryType *type)
{
	return VisitIn(static_cast<Type*>(type));
}

bool HierarchicalVisitor::VisitIn(EnumerationType *type)
{
	return VisitIn(static_cast<Type*>(type));
}

bool HierarchicalVisitor::VisitIn(TableType *type)
{
	return VisitIn(static_cast<Type*>(type));
}

bool HierarchicalVisitor::VisitIn(KeyedTableType *type)
{
	return VisitIn(static_cast<Type*>(type));
}

void HierarchicalVisitor::VisitOut(Type *type)
{
	VisitOut(static_cast<Node*>(type));
}

void HierarchicalVisitor::VisitOut(WildcardType *type)
{
	VisitOut(static_cast<Type*>(type));
}

void HierarchicalVisitor::VisitOut(BasicType *type)
{
	VisitOut(static_cast<Type*>(type));
}

void HierarchicalVisitor::VisitOut(FunctionType *type)
{
	VisitOut(static_cast<Type*>(type));
}

void HierarchicalVisitor::VisitOut(ListType *type)
{
	VisitOut(static_cast<Type*>(type));
}

void HierarchicalVisitor::VisitOut(DictionaryType *type)
{
	VisitOut(static_cast<Type*>(type));
}

void HierarchicalVisitor::VisitOut(EnumerationType *type)
{
	VisitOut(static_cast<Type*>(type));
}

void HierarchicalVisitor::VisitOut(TableType *type)
{
	VisitOut(static_cast<Type*>(type));
}

void HierarchicalVisitor::VisitOut(KeyedTableType *type)
{
	VisitOut(static_cast<Type*>(type));
}

}
