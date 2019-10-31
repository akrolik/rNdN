#pragma once

#include <string>
#include <sstream>
#include <vector>

#include "HorseIR/Traversal/ConstVisitor.h"

namespace HorseIR {

class PrettyPrinter : public ConstVisitor
{
public:
	static std::string PrettyString(const Node *node, bool quick = false);
	static std::string PrettyString(const Identifier *identifier, bool quick = false);

	// Modules

	void Visit(const Program *program) override;
	void Visit(const Module *module) override;
	void Visit(const ImportDirective *import) override;
	void Visit(const GlobalDeclaration *global) override;
	void Visit(const BuiltinFunction *function) override;
	void Visit(const Function *function) override;
	void Visit(const VariableDeclaration *declaration) override;

	// Statements

	void Visit(const DeclarationStatement *declarationS) override;
	void Visit(const AssignStatement *assignS) override;
	void Visit(const ExpressionStatement *expressionS) override;
        void Visit(const IfStatement *ifS) override;
	void Visit(const WhileStatement *whileS) override;
	void Visit(const RepeatStatement *repeatS) override;
	void Visit(const BlockStatement *blockS) override;
	void Visit(const ReturnStatement *returnS) override;
	void Visit(const BreakStatement *breakS) override;
	void Visit(const ContinueStatement *continueS) override;

	// Expressions

	void Visit(const CallExpression *call) override;
	void Visit(const CastExpression *cast) override;
	void Visit(const Identifier *identifier) override;

	// Literals

	void Visit(const BooleanLiteral *literal) override;
	void Visit(const CharLiteral *literal) override;
	void Visit(const Int8Literal *literal) override;
	void Visit(const Int16Literal *literal) override;
	void Visit(const Int32Literal *literal) override;
	void Visit(const Int64Literal *literal) override;
	void Visit(const Float32Literal *literal) override;
	void Visit(const Float64Literal *literal) override;
	void Visit(const ComplexLiteral *literal) override;
	void Visit(const StringLiteral *literal) override;
	void Visit(const SymbolLiteral *literal) override;
	void Visit(const DatetimeLiteral *literal) override;
	void Visit(const MonthLiteral *literal) override;
	void Visit(const DateLiteral *literal) override;
	void Visit(const MinuteLiteral *literal) override;
	void Visit(const SecondLiteral *literal) override;
	void Visit(const TimeLiteral *literal) override;
	void Visit(const FunctionLiteral *literal) override;

	// Types

	void Visit(const WildcardType *type) override;
	void Visit(const BasicType *type) override;
	void Visit(const FunctionType *type) override;
	void Visit(const ListType *type) override;
	void Visit(const DictionaryType *type) override;
	void Visit(const EnumerationType *type) override;
	void Visit(const TableType *type) override;
	void Visit(const KeyedTableType *type) override;

protected:
	std::stringstream m_string;
	unsigned int m_indent = 0;
	bool m_quick = false;

	void Indent();

	template<typename T>
	void CommaSeparated(const std::vector<T>& elements);

	template<typename T>
	void VectorLiteral(const std::vector<T>& values, bool boolean = false);
};

}
