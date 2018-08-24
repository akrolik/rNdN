#include "HorseIR/Analysis/SymbolTable.h"

#include "HorseIR/Tree/BuiltinMethod.h"
#include "HorseIR/Tree/Declaration.h"
#include "HorseIR/Tree/Import.h"
#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/MethodDeclaration.h"
#include "HorseIR/Tree/Module.h"
#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"
#include "HorseIR/Tree/Expressions/Literals/FunctionLiteral.h"
#include "HorseIR/Tree/Types/Type.h"

#include "Utils/Logger.h"

namespace HorseIR {

std::string SymbolTable::Entry::ToString() const
{
	std::string output;
	switch (kind)
	{
		case Kind::Module:
			output += "[module]";
			break;
		case Kind::Method:
			output += "[method] = " + static_cast<Method *>(node)->SignatureString();
			break;
		case Kind::Variable:
			output += "[variable] = " + node->ToString();
			break;
	}
	return output;
}

SymbolTable::Entry *SymbolTable::GetSymbol(const std::string& name)
{
	// Get symbol from the table, recursively traversing to parent
	// Order is:
	//  (1) Current scope
	//  (2) Imported identifiers
	//  (3) Parent scope

	if (m_table.find(name) != m_table.end())
	{
		return m_table.at(name);
	}
	if (m_imports.find(name) != m_imports.end())
	{
		return m_imports.at(name);
	}
	if (m_parent != nullptr)
	{
		return m_parent->GetSymbol(name);
	}
	return nullptr;
}

Module *SymbolTable::GetModule(const std::string& name)
{
	auto symbol = GetSymbol(name);
	if (symbol == nullptr)
	{
		Utils::Logger::LogError("Module " + name + " is not defined in current program");
	}
	else if (symbol->kind != SymbolTable::Entry::Kind::Module)
	{
		Utils::Logger::LogError("'" + name + "' is not a module");
	}
	return static_cast<Module *>(symbol->node);
}

MethodDeclaration *SymbolTable::GetMethod(const std::string& name)
{
	auto symbol = GetSymbol(name);
	if (symbol == nullptr)
	{
		Utils::Logger::LogError("Method '" + name + "' cannot be found in the current module scope");
	}
	else if (symbol->kind != SymbolTable::Entry::Kind::Method)
	{
		Utils::Logger::LogError("'" + name + "' is not a method");
	}
	return static_cast<MethodDeclaration *>(symbol->node);
}

Declaration *SymbolTable::GetVariable(const std::string& name)
{
	auto symbol = GetSymbol(name);
	if (symbol == nullptr)
	{
		Utils::Logger::LogError("Variable '" + name + "' cannot be found in the current method scope");
	}
	else if (symbol->kind != SymbolTable::Entry::Kind::Variable)
	{
		Utils::Logger::LogError("'" + name + "' is not a variable");
	}
	return static_cast<Declaration *>(symbol->node);
}

void SymbolTable::AddSymbol(const std::string& name, Entry *symbol)
{
	// Check if symbol is already defined at this scope

	if (m_table.find(name) != m_table.end())
	{
		Utils::Logger::LogError("Identifier " + name + " is already defined in current scope");
	}

	m_table.insert({name, symbol});
}

void SymbolTable::AddImport(const std::string& name, Entry *symbol)
{
	// Check if symbol is already defined at this scope

	if (m_imports.find(name) != m_imports.end())
	{
		Utils::Logger::LogError("Duplicate import for module '" + name + "'");
	}

	m_imports.insert({name, symbol});
}

void SymbolPass_Modules::Build(Program *program)
{
	auto symbolTable = new SymbolTable();
	program->SetSymbolTable(symbolTable);

	m_scopes.push(symbolTable);
	program->Accept(*this);
	m_scopes.pop();
}

void SymbolPass_Modules::Visit(Module *module)
{
	auto symbolTable = m_scopes.top();
	symbolTable->AddSymbol(module->GetName(), new SymbolTable::Entry(SymbolTable::Entry::Kind::Module, module));

	auto localSymbolTable = new SymbolTable(symbolTable);
	module->SetSymbolTable(localSymbolTable);

	m_scopes.push(localSymbolTable);
	ForwardTraversal::Visit(module);
	m_scopes.pop();
}

void SymbolPass_Modules::Visit(BuiltinMethod *method)
{
	auto symbolTable = m_scopes.top();
	symbolTable->AddSymbol(method->GetName(), new SymbolTable::Entry(SymbolTable::Entry::Kind::Method, method));
}

void SymbolPass_Modules::Visit(Method *method)
{
	auto symbolTable = m_scopes.top();
	symbolTable->AddSymbol(method->GetName(), new SymbolTable::Entry(SymbolTable::Entry::Kind::Method, method));
}

void SymbolPass_Imports::Build(Program *program)
{
	program->Accept(*this);
}

void SymbolPass_Imports::Visit(Module *module)
{
	m_module = module;
	ForwardTraversal::Visit(module);
	m_module = nullptr;
}

void SymbolPass_Imports::Visit(Import *import)
{
	auto symbolTable = m_module->GetSymbolTable();
	auto globalSymbolTable = symbolTable->GetParent();

	auto identifier = import->GetIdentifier();
	auto moduleName = identifier->GetModule();
	auto name = identifier->GetName();

	auto importedModule = globalSymbolTable->GetModule(moduleName);
	auto importedSymbolTable = importedModule->GetSymbolTable();

	if (name == "*")
	{
		for (auto& entry : importedSymbolTable->m_table)
		{
			symbolTable->AddImport(entry.first, entry.second);
		}
	}
	else
	{
		auto entry = importedSymbolTable->GetSymbol(name);
		symbolTable->AddImport(name, entry);
	}
}

void SymbolPass_Imports::Visit(Method *method)
{
	// Stop the traversal
}

void SymbolPass_Methods::Build(Program *program)
{
	program->Accept(*this);
}

void SymbolPass_Methods::Visit(Program *program)
{
	m_scopes.push(program->GetSymbolTable());
	ForwardTraversal::Visit(program);
	m_scopes.pop();
}

void SymbolPass_Methods::Visit(Module *module)
{
	m_scopes.push(module->GetSymbolTable());
	ForwardTraversal::Visit(module);
	m_scopes.pop();
}

void SymbolPass_Methods::Visit(Method *method)
{
	auto symbolTable = new SymbolTable(m_scopes.top());
	method->SetSymbolTable(symbolTable);
	m_scopes.push(symbolTable);
	ForwardTraversal::Visit(method);
	m_scopes.pop();
}

void SymbolPass_Methods::Visit(Declaration *declaration)
{
	auto symbolTable = m_scopes.top();
	symbolTable->AddSymbol(declaration->GetName(), new SymbolTable::Entry(SymbolTable::Entry::Kind::Variable, declaration));
	ForwardTraversal::Visit(declaration);
}

MethodDeclaration *SymbolPass_Methods::LookupIdentifier(const ModuleIdentifier *identifier)
{
	auto symbolTable = m_scopes.top();

	auto moduleName = identifier->GetModule();
	auto name = identifier->GetName();

	if (moduleName == "")
	{
		return symbolTable->GetMethod(name);
	}
	else
	{
		auto module = symbolTable->GetModule(moduleName);
		auto moduleSymbolTable = module->GetSymbolTable();
		return moduleSymbolTable->GetMethod(name);
	}
}

void SymbolPass_Methods::Visit(CallExpression *call)
{
	call->SetMethod(LookupIdentifier(call->GetIdentifier()));
	ForwardTraversal::Visit(call);
}

void SymbolPass_Methods::Visit(Identifier *identifier)
{
	auto symbolTable = m_scopes.top();

	auto name = identifier->GetString();
	auto declaration = symbolTable->GetVariable(name);
	identifier->SetDeclaration(declaration);
	ForwardTraversal::Visit(identifier);
}

void SymbolPass_Methods::Visit(FunctionLiteral *literal)
{
	auto function = LookupIdentifier(literal->GetIdentifier());
	literal->SetMethod(function);
	ForwardTraversal::Visit(literal);
}

void SymbolTableBuilder::Build(Program *program)
{
	// Pass 1: Collect modules and methods

	SymbolPass_Modules modules;
	modules.Build(program);

	// Pass 2: Collect imports

	SymbolPass_Imports imports;
	imports.Build(program);

	// Pass 3: Method bodies

	SymbolPass_Methods methods;
	methods.Build(program);
}

void SymbolTableDumper::Dump(const Program *program)
{
	program->Accept(*this);
}

void SymbolTableDumper::Visit(const Program *program)
{
	Utils::Logger::LogInfo("Global symbol table");

	auto symbolTable = program->GetSymbolTable();
	for (const auto& entry : symbolTable->m_table)
	{
		Utils::Logger::LogInfo(entry.first + " " + entry.second->ToString(), 1);
	}

	m_scopes.push(symbolTable);
	ConstForwardTraversal::Visit(program);
	m_scopes.pop();
}

void SymbolTableDumper::Visit(const Module *module)
{
	Utils::Logger::LogInfo();
	Utils::Logger::LogInfo("Module symbol table: " + module->GetName());

	auto symbolTable = module->GetSymbolTable();
	for (const auto& entry : symbolTable->m_table)
	{
		Utils::Logger::LogInfo(entry.first + " " + entry.second->ToString(), 1);
	}
	if (symbolTable->m_imports.size() > 0)
	{
		Utils::Logger::LogInfo();
		Utils::Logger::LogInfo(" Imports");
	}
	for (const auto& entry : symbolTable->m_imports)
	{
		Utils::Logger::LogInfo(" * " + entry.first + " " + entry.second->ToString());
	}

	m_scopes.push(symbolTable);
	ConstForwardTraversal::Visit(module);
	m_scopes.pop();
}

void SymbolTableDumper::Visit(const Method *method)
{
	Utils::Logger::LogInfo();
	Utils::Logger::LogInfo("Method symbol table: " + method->GetName());

	m_scopes.push(method->GetSymbolTable());
	ConstForwardTraversal::Visit(method);
	m_scopes.pop();
}

void SymbolTableDumper::Visit(const Declaration *declaration)
{
	auto symbolTable = m_scopes.top();
	auto entry = symbolTable->GetSymbol(declaration->GetName());

	Utils::Logger::LogInfo(declaration->GetName() + " " + entry->ToString(), 1);

	ConstForwardTraversal::Visit(declaration);
}

}
