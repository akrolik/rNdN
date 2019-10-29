#include "HorseIR/Semantics/SymbolTable/SymbolTableBuilder.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace HorseIR {

void SymbolTableBuilder::Build(Program *program)
{
	// Pass 1: Collect modules and functions

	SymbolPass_Modules modules;
	modules.Build(program);

	// Pass 2: Collect imports

	SymbolPass_Imports imports;
	imports.Build(program);

	// Pass 3: Function bodies

	SymbolPass_Functions functions;
	functions.Build(program);
}

void SymbolPass_Modules::Build(Program *program)
{
	program->Accept(*this);
}

bool SymbolPass_Modules::VisitIn(Program *program)
{
	auto symbolTable = new SymbolTable();
	program->SetSymbolTable(symbolTable);

	m_currentSymbolTable = symbolTable;
	return true;
}

void SymbolPass_Modules::VisitOut(Program *program)
{
	m_currentSymbolTable = m_currentSymbolTable->GetParent();
}

bool SymbolPass_Modules::VisitIn(Module *module)
{
	auto& name = module->GetName();
	m_currentSymbolTable->AddSymbol(
		name,
		new SymbolTable::Symbol(SymbolTable::Symbol::Kind::Module, name, module)
	);

	auto localSymbolTable = new SymbolTable(m_currentSymbolTable);
	module->SetSymbolTable(localSymbolTable);

	m_currentSymbolTable = localSymbolTable;
	return true;
}

void SymbolPass_Modules::VisitOut(Module *module)
{
	m_currentSymbolTable = m_currentSymbolTable->GetParent();
}

bool SymbolPass_Modules::VisitIn(GlobalDeclaration *global)
{
	auto declaration = global->GetDeclaration();
	auto& name = declaration->GetName();
	auto symbol = new SymbolTable::Symbol(SymbolTable::Symbol::Kind::Variable, name, declaration);
	m_currentSymbolTable->AddSymbol(name, symbol);
	declaration->SetSymbol(symbol);
	return false;
}

bool SymbolPass_Modules::VisitIn(FunctionDeclaration *function)
{
	auto& name = function->GetName();
	m_currentSymbolTable->AddSymbol(
		name,
		new SymbolTable::Symbol(SymbolTable::Symbol::Kind::Function, name, function)
	);
	return false;
}

void SymbolPass_Imports::Build(Program *program)
{
	program->Accept(*this);
}

bool SymbolPass_Imports::VisitIn(Program *program)
{
	m_globalSymbolTable = program->GetSymbolTable();
	return true;
}

void SymbolPass_Imports::VisitOut(Program *program)
{
	m_globalSymbolTable = nullptr;
}

bool SymbolPass_Imports::VisitIn(Module *module)
{
	auto importTable = new SymbolTable();
	module->GetSymbolTable()->SetImportTable(importTable);

	m_currentImportTable = importTable;
	return true;
}

void SymbolPass_Imports::VisitOut(Module *module)
{
	m_currentImportTable = nullptr;
}

bool SymbolPass_Imports::VisitIn(ImportDirective *import)
{
	auto moduleName = import->GetModuleName();

	auto importedModule = m_globalSymbolTable->GetModule(moduleName);
	auto importedSymbolTable = importedModule->GetSymbolTable();

	for (auto& name : import->GetContents())
	{
		if (name == "*")
		{
			for (auto& [symbolName, symbol] : importedSymbolTable->m_table)
			{
				m_currentImportTable->AddSymbol(symbolName, symbol, true);
			}
		}
		else
		{
			auto symbol = importedSymbolTable->GetSymbol(name);
			m_currentImportTable->AddSymbol(name, symbol, true);
		}
	}
	return false;
}

bool SymbolPass_Imports::VisitIn(ModuleContent *content)
{
	// Stop the traversal
	return false;
}

void SymbolPass_Functions::Build(Program *program)
{
	program->Accept(*this);
}

bool SymbolPass_Functions::VisitIn(Program *program)
{
	m_currentSymbolTable = program->GetSymbolTable();
	return true;
}

void SymbolPass_Functions::VisitOut(Program *program)
{
	m_currentSymbolTable = m_currentSymbolTable->GetParent();
}


bool SymbolPass_Functions::VisitIn(Module *module)
{
	m_currentSymbolTable = module->GetSymbolTable();
	return true;
}

void SymbolPass_Functions::VisitOut(Module *module)
{
	m_currentSymbolTable = m_currentSymbolTable->GetParent();
}

bool SymbolPass_Functions::VisitIn(ModuleContent *content)
{
	// Stop the traversal
	return false;
}

bool SymbolPass_Functions::VisitIn(Function *function)
{
	auto symbolTable = new SymbolTable(m_currentSymbolTable);
	function->SetSymbolTable(symbolTable);
	
	m_currentSymbolTable = symbolTable;
	return true;
}

void SymbolPass_Functions::VisitOut(Function *function)
{
	m_currentSymbolTable = m_currentSymbolTable->GetParent();
}

bool SymbolPass_Functions::VisitIn(BlockStatement *blockS)
{
	auto symbolTable = new SymbolTable(m_currentSymbolTable);
	blockS->SetSymbolTable(symbolTable);
	
	m_currentSymbolTable = symbolTable;
	return true;
}

void SymbolPass_Functions::VisitOut(BlockStatement *blockS)
{
	m_currentSymbolTable = m_currentSymbolTable->GetParent();
}

bool SymbolPass_Functions::VisitIn(AssignStatement *assignS)
{
	// Invert the order of child traversal

	assignS->GetExpression()->Accept(*this);
	for (auto& target : assignS->GetTargets())
	{
		target->Accept(*this);
	}
	return false;
}

bool SymbolPass_Functions::VisitIn(VariableDeclaration *declaration)
{
	auto& name = declaration->GetName();
	auto symbol = new SymbolTable::Symbol(SymbolTable::Symbol::Kind::Variable, name, declaration);
	declaration->SetSymbol(symbol);
	m_currentSymbolTable->AddSymbol(name, symbol); 
	return true;
}

bool SymbolPass_Functions::VisitIn(Identifier *identifier)
{
	auto symbol = LookupIdentifier(identifier);
	identifier->SetSymbol(symbol);
	return true;
}

SymbolTable::Symbol *SymbolPass_Functions::LookupIdentifier(const Identifier *identifier)
{
	auto moduleName = identifier->GetModule();
	auto name = identifier->GetName();

	if (moduleName == "")
	{
		return m_currentSymbolTable->GetSymbol(name);
	}
	else
	{
		auto module = m_currentSymbolTable->GetModule(moduleName);
		auto moduleSymbolTable = module->GetSymbolTable();
		return moduleSymbolTable->GetSymbol(name);
	}
}

void SymbolPass_Functions::VisitOut(FunctionLiteral *literal)
{
	auto symbol = literal->GetIdentifier()->GetSymbol();
	if (symbol->kind != SymbolTable::Symbol::Kind::Function)
	{
		Utils::Logger::LogError("'" + PrettyPrinter::PrettyString(static_cast<Operand*>(literal->GetIdentifier())) + "' is not a function");
	}

	const auto function = static_cast<const FunctionDeclaration *>(symbol->node);
	literal->SetFunction(function);
}

}
