#include "PTX/Utils/PrettyPrinter.h"

#include "PTX/Tree/Tree.h"

#include "Utils/Logger.h"
#include "Utils/String.h"

namespace PTX {

std::string PrettyPrinter::PrettyString(const Node *node, bool quick)
{
	PrettyPrinter printer;
	printer.m_string.str("");
	printer.m_quick = quick;
	node->Accept(printer);
	return printer.m_string.str();
}

void PrettyPrinter::Indent()
{
	m_string << std::string(m_indent * Utils::Logger::IndentSize, ' ');
}

void PrettyPrinter::Visit(const Program *program)
{
	auto first = true;
	for (const auto& module : program->GetModules())
	{
		if (!first)
		{
			m_string << std::endl << std::endl;
		}
		first = false;
		module->Accept(*this);
	}
}

void PrettyPrinter::Visit(const Module *module)
{
	// Header

	m_string << "//" << std::endl;
	m_string << "// Generated by r3r3 compiler" << std::endl;
	m_string << "// version 0.1" << std::endl;
	m_string << "//" << std::endl << std::endl;

	// Metadata

	m_string << ".version " << module->GetMajorVersion() << "." << module->GetMinorVersion() << std::endl;
	m_string << ".target " << module->GetTarget() << std::endl;
	m_string << ".address_size " << DynamicBitSize::GetBits(module->GetAddressSize()) << std::endl;

	// Directives

	for (const auto& directive : module->GetDirectives())
	{
		directive->Accept(*this);
		m_string << std::endl;
	}

	auto first = true;
	for (const auto& declaration : module->GetDeclarations())
	{
		m_string << std::endl;
		if (!first)
		{
			m_string << std::endl;
		}
		first = false;
		declaration->Accept(*this);
	}
}

void PrettyPrinter::Visit(const BasicBlock *block)
{
	m_string << std::endl;
	block->GetLabel()->Accept(*this);
	m_string << ":" << std::endl;

	auto quick = m_quick;
	m_quick = true;
	for (const auto& statement : block->GetStatements())
	{
		statement->Accept(*this);
		m_string << std::endl;
	}
	m_quick = quick;
}

void PrettyPrinter::Visit(const FunctionDeclaration<VoidType> *function)
{
	// Linking
	auto linkDirective = function->GetLinkDirective();
	if (linkDirective != Function::LinkDirective::None)
	{
		m_string << Function::LinkDirectiveString(linkDirective) + " ";
	}
	m_string << function->GetDirectives() + " ";

	// Return
	if (const auto returnDeclaration = function->GetReturnDeclaration())
	{
		m_string << "(";
		returnDeclaration->Accept(*this);
		m_string << ")";
	}
	m_string << function->GetName();
	
	// Parameters
	auto first = true;
	m_string << "(";
	m_indent++;
	for (const auto& parameter : function->GetParameters())
	{
		if (!first)
		{
			m_string << ",";
		}
		first = false;

		m_string << std::endl;
		Indent();
		parameter->Accept(*this);
	}
	if (!first)
	{
		m_string << std::endl;
	}
	m_indent--;
	Indent();
	m_string << ")";

	// Options
	const auto& options = function->GetOptions();
	if (options.GetBlockSize() != FunctionOptions::DynamicBlockSize)
	{
		m_string << " .reqntid " + std::to_string(options.GetBlockSize());
	}

	if (!m_definition)
	{
		m_string << ";";
	}
}

void PrettyPrinter::Visit(const FunctionDefinition<VoidType> *function)
{
	m_definition = true;
	ConstVisitor::Visit(function);
	m_definition = false;

	m_string << "{" << std::endl;
	m_indent++;
	if (const auto cfg = function->GetControlFlowGraph())
	{
		cfg->LinearOrdering([&](Analysis::ControlFlowNode& block)
		{
			block->Accept(*this);
		});
	}
	else
	{
		for (const auto& statement : function->GetStatements())
		{
			statement->Accept(*this);
			m_string << std::endl;
		}
	}
	m_indent--;
	m_string << "}";
}

void PrettyPrinter::Visit(const VariableDeclaration *declaration)
{
	m_string << declaration->ToString();
}

void PrettyPrinter::Visit(const FileDirective *directive)
{
	Indent();
	m_string << ".file " << directive->GetIndex() << " \"" << directive->GetName() << "\"";

	auto timestamp = directive->GetTimestamp();
	auto filesize = directive->GetFilesize();
	if (timestamp > 0 || filesize > 0)
	{
		m_string << ", " << timestamp << ", " << filesize;
	}
}

void PrettyPrinter::Visit(const LocationDirective *directive)
{
	Indent();
	m_string << ".loc " << directive->GetFile()->GetIndex() << " " << directive->GetLine() << " " << directive->GetColumn();
}

void PrettyPrinter::Visit(const BlockStatement *block)
{
	Indent();
	m_string << "{" << std::endl;
	m_indent++;
	for (const auto& statement : block->GetStatements())
	{
		statement->Accept(*this);
		m_string << std::endl;
	}
	m_indent--;
	Indent();
	m_string << "}";
}

void PrettyPrinter::Visit(const CommentStatement *statement)
{
	std::string indentString(8 * m_indent, ' ');
	if (statement->IsMultiline())
	{
		m_string << indentString << "/*" << std::endl << indentString << " * ";
		m_string << Utils::String::ReplaceString(statement->GetComment(), "\n", "\n" + indentString + " * ");
		m_string << indentString + " */";
	}
	m_string << indentString << "// " << Utils::String::ReplaceString(statement->GetComment(), "\n", "\n" + indentString + "// ");
}

void PrettyPrinter::Visit(const DeclarationStatement *statement)
{
	Indent();
	statement->GetDeclaration()->Accept(*this);
	m_string << ";";
}

void PrettyPrinter::Visit(const InstructionStatement *statement)
{
	auto prefix = statement->GetPrefix();
	if (prefix.size() > 0)
	{
		// Custom indent if instruction prefix

		int padding = 8 * m_indent - 1 - prefix.size();
		if (padding > 0)
		{
			m_string << std::string(padding, ' ');
		}
		m_string << prefix << " ";
	}
	else
	{
		Indent();
	}

	m_string << statement->GetOpCode();
	auto first = true;
	for (const auto& operand : statement->GetOperands())
	{
		if (first)
		{
			m_string << " ";
			first = false;
		}
		else
		{
			m_string << ", ";
		}
		operand->Accept(*this);
	}
	m_string << ";";

	if (!m_quick && dynamic_cast<const BranchInstruction *>(statement))
	{
		m_string << std::endl;
	}
}

void PrettyPrinter::Visit(const LabelStatement *statement)
{
	// No indent

	m_string << std::endl;
	statement->GetLabel()->Accept(*this);
	m_string << ":";
}

void PrettyPrinter::Visit(const Operand *operand)
{
	m_string << operand->ToString();
}

}
