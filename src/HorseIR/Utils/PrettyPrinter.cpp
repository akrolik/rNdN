#include "HorseIR/Utils/PrettyPrinter.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {

std::string PrettyPrinter::PrettyString(const Node *node, bool quick)
{
	PrettyPrinter printer;
	printer.m_string.str("");
	printer.m_quick = quick;
	node->Accept(printer);
	return printer.m_string.str();
}
 
std::string PrettyPrinter::PrettyString(const Identifier *identifier, bool quick)
{
	PrettyPrinter printer;
	printer.m_string.str("");
	printer.m_quick = quick;
	identifier->Accept(printer);
	return printer.m_string.str();
}
 
void PrettyPrinter::Indent()
{
	for (unsigned int i = 0; i < m_indent; ++i)
	{
		m_string << "\t";
	}
}

template<typename T>
void PrettyPrinter::CommaSeparated(const std::vector<T>& elements)
{
	bool first = true;
	for (const auto& element : elements)
	{
		if (!first)
		{
			m_string << ", ";
		}
		first = false;
		element->Accept(*this);
	}
}

template<>
void PrettyPrinter::CommaSeparated<std::string>(const std::vector<std::string>& elements)
{
	bool first = true;
	for (const auto& element : elements)
	{
		if (!first)
		{
			m_string << ", ";
		}
		first = false;
		m_string << element;
	}
}

void PrettyPrinter::Visit(const Program *program)
{
	bool first = true;
	for (const auto& module : program->GetModules())
	{
		if (!first)
		{
			m_string << std::endl;
		}
		first = false;
		module->Accept(*this);
	}
}

void PrettyPrinter::Visit(const Module *module)
{
	m_string << "module " << module->GetName() << " {" << std::endl;
	m_indent++;

	for (const auto& moduleContent : module->GetContents())
	{
		moduleContent->Accept(*this);
	}

	m_indent--;
	m_string << "}";
}

void PrettyPrinter::Visit(const ImportDirective *import)
{
	Indent();
	m_string << "import " << import->GetModuleName() << ".";

	const auto& contents = import->GetContents();
	if (contents.size() == 1)
	{
		m_string << contents.at(0);
	}
	else
	{
		m_string << "{";
		for (const auto& content : contents)
		{
			m_string << content;
		}
		m_string << "}";
	}
	m_string << ";" << std::endl;
}

void PrettyPrinter::Visit(const GlobalDeclaration *global)
{
	Indent();
	m_string << "global ";
	global->GetDeclaration()->Accept(*this);
	m_string << " = ";
	global->GetExpression()->Accept(*this);
	m_string << ";" << std::endl;
}

void PrettyPrinter::Visit(const BuiltinFunction *function)
{
	Indent();
	m_string << "def " << function->GetName() << "() __BUILTIN__";
	if (m_quick)
	{
		return;
	}
	m_string << std::endl;
}

void PrettyPrinter::Visit(const Function *function)
{
	Indent();
	m_string << std::string((function->IsKernel()) ? "kernel " : "def ") << function->GetName() << "(";
	CommaSeparated(function->GetParameters());
	m_string << ")";

	const auto& returnTypes = function->GetReturnTypes();
	if (returnTypes.size() > 0)
	{
		m_string << " : ";
		CommaSeparated(returnTypes);
	}

	// Quick printing excludes body
	if (m_quick)
	{
		return;
	}

	m_string << " {" << std::endl;

	m_indent++;
	for (const auto& statement : function->GetStatements())
	{
		statement->Accept(*this);
	}
	m_indent--;

	Indent();
	m_string << "}" << std::endl;
}

void PrettyPrinter::Visit(const VariableDeclaration *declaration)
{
	m_string << declaration->GetName() << ":";
	declaration->GetType()->Accept(*this);
}

void PrettyPrinter::Visit(const DeclarationStatement *declarationS)
{
	Indent();
	m_string << "var ";
	declarationS->GetDeclaration()->Accept(*this);
	if (m_quick)
	{
		return;
	}
	m_string << ";" << std::endl;
}

void PrettyPrinter::Visit(const AssignStatement *assignS)
{
	Indent();
	CommaSeparated(assignS->GetTargets());

	m_string << " = ";
	assignS->GetExpression()->Accept(*this);
	if (m_quick)
	{
		return;
	}
	m_string << ";" << std::endl;
}

void PrettyPrinter::Visit(const ExpressionStatement *expressionS)
{
	Indent();
	expressionS->GetExpression()->Accept(*this);
	if (m_quick)
	{
		return;
	}
	m_string << ";" << std::endl;
}

void PrettyPrinter::Visit(const IfStatement *ifS)
{
	Indent();
	m_string << "if (";
	ifS->GetCondition()->Accept(*this);
	m_string << ")";
	if (m_quick)
	{
		return;
	}
	m_string << " ";
	ifS->GetTrueBlock()->Accept(*this);

	if (ifS->HasElseBranch())
	{
		m_string << " else ";
		ifS->GetElseBlock()->Accept(*this);
	}

	m_string << std::endl;
}

void PrettyPrinter::Visit(const WhileStatement *whileS)
{
	Indent();
	m_string << "while (";
	whileS->GetCondition()->Accept(*this);
	m_string << ")";
	if (m_quick)
	{
		return;
	}
	m_string << " ";
	whileS->GetBody()->Accept(*this);
	m_string << std::endl;
}

void PrettyPrinter::Visit(const RepeatStatement *repeatS)
{
	Indent();
	m_string << "repeat (";
	repeatS->GetCondition()->Accept(*this);
	m_string << ")";
	if (m_quick)
	{
		return;
	}
	m_string << " ";
	repeatS->GetBody()->Accept(*this);
	m_string << std::endl;
}

void PrettyPrinter::Visit(const BlockStatement *blockS)
{
	m_string << "{" << std::endl;

	m_indent++;
	for (const auto& statement : blockS->GetStatements())
	{
		statement->Accept(*this);
	}
	m_indent--;

	Indent();
	m_string << "}";
}

void PrettyPrinter::Visit(const ReturnStatement *returnS)
{
	Indent();
	m_string << "return";

	if (returnS->GetOperandsCount() > 0)
	{
		m_string << " ";
		CommaSeparated(returnS->GetOperands());
	}
	if (m_quick)
	{
		return;
	}
	m_string << ";" << std::endl;
}

void PrettyPrinter::Visit(const BreakStatement *breakS)
{
	Indent();
	m_string << "break;" << std::endl;
}

void PrettyPrinter::Visit(const ContinueStatement *continueS)
{
	Indent();
	m_string << "continue;" << std::endl;
}

void PrettyPrinter::Visit(const CallExpression *call)
{
	call->GetFunctionLiteral()->Accept(*this);
	m_string << "(";
	CommaSeparated(call->GetArguments());
	m_string << ")";
}

void PrettyPrinter::Visit(const CastExpression *cast)
{
	m_string << "check_cast(";
	cast->GetExpression()->Accept(*this);
	m_string << ", ";
	cast->GetCastType()->Accept(*this);
	m_string << ")";
}

void PrettyPrinter::Visit(const Identifier *identifier)
{
	if (identifier->HasModule())
	{
		m_string << identifier->GetModule() << ".";
	}
	m_string << identifier->GetName();
}

template<typename T>
void PrettyPrinter::VectorLiteral(const std::vector<T>& values, bool boolean)
{
	if (values.size() > 1)
	{
		m_string << "(";
	}

	bool first = true;
	for (const auto& value : values)
	{
		if (!first)
		{
			m_string << ", ";
		}
		first = false;
		if constexpr(std::is_pointer<T>::value)
		{
			m_string << *value;
		}
		else
		{
			if (boolean)
			{
				if constexpr(std::is_same<T, std::string>::value)
				{
					m_string << value;
				}
				else
				{
					m_string << ((value) ? "1" : "0");
				}
			}
			else
			{
				m_string << value;
			}
		}
	}

	if (values.size() > 1)
	{
		m_string << ")";
	}
}

void PrettyPrinter::Visit(const BooleanLiteral *literal)
{
	VectorLiteral(literal->GetValues(), true);
	m_string << ":" << BasicType::BasicKindString(BasicType::BasicKind::Boolean);
}

void PrettyPrinter::Visit(const CharLiteral *literal)
{
	VectorLiteral(literal->GetValues());
	m_string << ":" << BasicType::BasicKindString(BasicType::BasicKind::Char);
}

void PrettyPrinter::Visit(const Int8Literal *literal)
{
	VectorLiteral(literal->GetValues());
	m_string << ":" << BasicType::BasicKindString(BasicType::BasicKind::Int8);
}

void PrettyPrinter::Visit(const Int16Literal *literal)
{
	VectorLiteral(literal->GetValues());
	m_string << ":" << BasicType::BasicKindString(BasicType::BasicKind::Int16);
}

void PrettyPrinter::Visit(const Int32Literal *literal)
{
	VectorLiteral(literal->GetValues());
	m_string << ":" << BasicType::BasicKindString(BasicType::BasicKind::Int32);
}

void PrettyPrinter::Visit(const Int64Literal *literal)
{
	VectorLiteral(literal->GetValues());
	m_string << ":" << BasicType::BasicKindString(BasicType::BasicKind::Int64);
}

void PrettyPrinter::Visit(const Float32Literal *literal)
{
	VectorLiteral(literal->GetValues());
	m_string << ":" << BasicType::BasicKindString(BasicType::BasicKind::Float32);
}

void PrettyPrinter::Visit(const Float64Literal *literal)
{
	VectorLiteral(literal->GetValues());
	m_string << ":" << BasicType::BasicKindString(BasicType::BasicKind::Float64);
}

void PrettyPrinter::Visit(const ComplexLiteral *literal)
{
	VectorLiteral(literal->GetValues());
	m_string << ":" << BasicType::BasicKindString(BasicType::BasicKind::Complex);
}

void PrettyPrinter::Visit(const StringLiteral *literal)
{
	VectorLiteral(literal->GetValues());
	m_string << ":" << BasicType::BasicKindString(BasicType::BasicKind::String);
}

void PrettyPrinter::Visit(const SymbolLiteral *literal)
{
	VectorLiteral(literal->GetValues());
	m_string << ":" << BasicType::BasicKindString(BasicType::BasicKind::Symbol);
}

void PrettyPrinter::Visit(const DatetimeLiteral *literal)
{
	VectorLiteral(literal->GetValues());
	m_string << ":" << BasicType::BasicKindString(BasicType::BasicKind::Datetime);
}

void PrettyPrinter::Visit(const MonthLiteral *literal)
{
	VectorLiteral(literal->GetValues());
	m_string << ":" << BasicType::BasicKindString(BasicType::BasicKind::Month);
}

void PrettyPrinter::Visit(const DateLiteral *literal)
{
	VectorLiteral(literal->GetValues());
	m_string << ":" << BasicType::BasicKindString(BasicType::BasicKind::Date);
}

void PrettyPrinter::Visit(const MinuteLiteral *literal)
{
	VectorLiteral(literal->GetValues());
	m_string << ":" << BasicType::BasicKindString(BasicType::BasicKind::Minute);
}

void PrettyPrinter::Visit(const SecondLiteral *literal)
{
	VectorLiteral(literal->GetValues());
	m_string << ":" << BasicType::BasicKindString(BasicType::BasicKind::Second);
}

void PrettyPrinter::Visit(const TimeLiteral *literal)
{
	VectorLiteral(literal->GetValues());
	m_string << ":" << BasicType::BasicKindString(BasicType::BasicKind::Time);
}

void PrettyPrinter::Visit(const FunctionLiteral *literal)
{
	m_string << "@";
	literal->GetIdentifier()->Accept(*this);
}

void PrettyPrinter::Visit(const BasicType *type)
{
	m_string << BasicType::BasicKindString(type->GetBasicKind());
}

void PrettyPrinter::Visit(const FunctionType *type)
{
	m_string << "func";
}

void PrettyPrinter::Visit(const ListType *type)
{
	m_string << "list<";
	CommaSeparated(type->GetElementTypes());
	m_string << ">";
}

void PrettyPrinter::Visit(const DictionaryType *type)
{
	m_string << "dict<";
	type->GetKeyType()->Accept(*this);
	m_string << ", ";
	type->GetValueType()->Accept(*this);
	m_string << ">";
}

void PrettyPrinter::Visit(const EnumerationType *type)
{
	m_string << "enum<";
	type->GetElementType()->Accept(*this);
	m_string << ">";
}

void PrettyPrinter::Visit(const TableType *type)
{
	m_string << "table";
}

void PrettyPrinter::Visit(const KeyedTableType *type)
{
	m_string << "ktable";
}

}
