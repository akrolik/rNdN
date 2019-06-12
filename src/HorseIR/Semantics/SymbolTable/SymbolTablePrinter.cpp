#include "HorseIR/Semantics/SymbolTable/SymbolTablePrinter.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {

void SymbolTablePrinter::Indent()
{
	for (unsigned int i = 0; i < m_indent; ++i)
	{
		m_string << "\t";
	}
}

std::string SymbolTablePrinter::PrettyString(const Program *program)
{
	SymbolTablePrinter printer;
	printer.m_string.str("");
	program->Accept(printer);
	return printer.m_string.str();
}

bool SymbolTablePrinter::VisitIn(const Program *program)
{
	m_string << "Global symbol table" << std::endl;

	auto symbolTable = program->GetSymbolTable();
	m_indent++;
	for (const auto& symbol : symbolTable->m_table)
	{
		Indent();
		m_string << symbol.first << " " << *symbol.second << std::endl;
	}

	m_currentSymbolTable = symbolTable;
	return true;
}

void SymbolTablePrinter::VisitOut(const Program *program)
{
	m_indent--;
	m_currentSymbolTable = m_currentSymbolTable->GetParent();
}

bool SymbolTablePrinter::VisitIn(const Module *module)
{
	m_string << std::endl;
	Indent();
	m_string << "Module symbol table: " << module->GetName() << std::endl;
	m_indent++;

	auto symbolTable = module->GetSymbolTable();
	auto importTable = symbolTable->GetImportTable();

	bool hasImports = (importTable != nullptr && importTable->m_table.size() > 0);
	if (hasImports)
	{
		Indent();
		m_string << "Imports" << std::endl;
	}
	for (const auto& symbol : importTable->m_table)
	{
		Indent();
		m_string << " * " << symbol.first << " " << *symbol.second << std::endl;
	}

	if (hasImports)
	{
		m_string << std::endl;
	}
	for (const auto& symbol : symbolTable->m_table)
	{
		Indent();
		m_string << symbol.first << " " << *symbol.second << std::endl;
	}

	m_currentSymbolTable = symbolTable;
	return true;
}

void SymbolTablePrinter::VisitOut(const Module *module)
{
	m_indent--;
	m_currentSymbolTable = m_currentSymbolTable->GetParent();
}

bool SymbolTablePrinter::VisitIn(const Function *function)
{
	m_string << std::endl;

	Indent();
	m_string << "Function symbol table: " << function->GetName() << std::endl;
	m_indent++;

	m_currentSymbolTable = function->GetSymbolTable();
	return true;
}

void SymbolTablePrinter::VisitOut(const Function *function)
{
	m_indent--;
	m_currentSymbolTable = m_currentSymbolTable->GetParent();
}

bool SymbolTablePrinter::VisitIn(const BlockStatement *blockS)
{
	m_string << std::endl;

	Indent();
	m_string << "Block symbol table" << std::endl;
	m_indent++;

	m_currentSymbolTable = blockS->GetSymbolTable();
	return true;
}

void SymbolTablePrinter::VisitOut(const BlockStatement *blockS)
{
	m_indent--;
	m_currentSymbolTable = m_currentSymbolTable->GetParent();
}

bool SymbolTablePrinter::VisitIn(const VariableDeclaration *declaration)
{
	auto symbol = m_currentSymbolTable->GetSymbol(declaration->GetName());
	Indent();
	m_string << declaration->GetName() << " " << *symbol << std::endl;
	return true;
}                

}
