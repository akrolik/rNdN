#include "HorseIR/SymbolTable.h"

#include "HorseIR/Tree/BuiltinMethod.h"
#include "HorseIR/Tree/Import.h"
#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/MethodDeclaration.h"
#include "HorseIR/Tree/Module.h"
#include "HorseIR/Tree/Parameter.h"
#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"
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

void SymbolTable::Insert(const std::string& name, Entry *symbol)
{
	// Check if symbol is already defined at this scope

	if (m_table.find(name) != m_table.end())
	{
		Utils::Logger::LogError("Identifier " + name + " is already defined in scope");
	}

	m_table.insert({name, symbol});
}

SymbolTable::Entry *SymbolTable::Get(const std::string& name)
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
		return m_parent->Get(name);
	}
	return nullptr;
}

Module *SymbolTable::GetModule(const std::string& name)
{
	auto symbol = Get(name);
	if (symbol == nullptr)
	{
		Utils::Logger::LogError("Module " + name + " is not defined in program");
	}
	else if (symbol->kind != SymbolTable::Entry::Kind::Module)
	{
		Utils::Logger::LogError(name + " is not a module");
	}
	return static_cast<Module *>(symbol->node);
}

MethodDeclaration *SymbolTable::GetMethod(const std::string& name)
{
	auto symbol = Get(name);
	if (symbol == nullptr)
	{
		Utils::Logger::LogError("Method " + name + " is not defined in module");
	}
	else if (symbol->kind != SymbolTable::Entry::Kind::Method)
	{
		Utils::Logger::LogError(name + " is not a method");
	}
	return static_cast<MethodDeclaration *>(symbol->node);
}

Type *SymbolTable::GetVariable(const std::string& name)
{
	auto symbol = Get(name);
	if (symbol == nullptr)
	{
		Utils::Logger::LogError("Variable " + name + " is not defined in method");
	}
	else if (symbol->kind != SymbolTable::Entry::Kind::Variable)
	{
		Utils::Logger::LogError(name + " is not a variable");
	}
	return static_cast<Type *>(symbol->node);
}

void SymbolTable::AddImport(const std::string& name, Entry *symbol)
{
	// Check if symbol is already defined at this scope

	if (m_imports.find(name) != m_imports.end())
	{
		Utils::Logger::LogError("Identifier " + name + " is already imported in module");
	}

	m_imports.insert({name, symbol});
}

std::string SymbolTable::ToString() const
{
	std::string output;
	for (const auto& entry : m_table)
	{
		output += " - " + entry.first + " " + entry.second->ToString() + "\n";
	}

	for (const auto& entry : m_imports)
	{
		output += " * " + entry.first + " " + entry.second->ToString() + "\n";
	}
	return output;
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
	symbolTable->Insert(module->GetName(), new SymbolTable::Entry(SymbolTable::Entry::Kind::Module, module));

	auto localSymbolTable = new SymbolTable(symbolTable);
	module->SetSymbolTable(localSymbolTable);

	m_scopes.push(localSymbolTable);
	ForwardTraversal::Visit(module);
	m_scopes.pop();
}

void SymbolPass_Modules::Visit(BuiltinMethod *method)
{
	auto symbolTable = m_scopes.top();
	symbolTable->Insert(method->GetName(), new SymbolTable::Entry(SymbolTable::Entry::Kind::Method, method));
}


void SymbolPass_Modules::Visit(Method *method)
{
	auto symbolTable = m_scopes.top();
	symbolTable->Insert(method->GetName(), new SymbolTable::Entry(SymbolTable::Entry::Kind::Method, method));
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
		auto entry = importedSymbolTable->Get(name);
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

void SymbolPass_Methods::Visit(Parameter *parameter)
{
	auto symbolTable = m_scopes.top();
	symbolTable->Insert(parameter->GetName(), new SymbolTable::Entry(SymbolTable::Entry::Kind::Variable, parameter->GetType()));
	ForwardTraversal::Visit(parameter);
}

void SymbolPass_Methods::Visit(AssignStatement *assign)
{
	auto symbolTable = m_scopes.top();
	symbolTable->Insert(assign->GetTargetName(), new SymbolTable::Entry(SymbolTable::Entry::Kind::Variable, assign->GetType()));
	ForwardTraversal::Visit(assign);
}

void SymbolPass_Methods::Visit(CallExpression *call)
{
	auto symbolTable = m_scopes.top();

	auto identifier = call->GetIdentifier();
	auto moduleName = identifier->GetModule();
	auto name = identifier->GetName();

	if (moduleName == "")
	{
		auto method = symbolTable->GetMethod(name);
		call->SetMethod(method);
	}
	else
	{
		auto module = symbolTable->GetModule(moduleName);
		auto moduleSymbolTable = module->GetSymbolTable();

		auto method = moduleSymbolTable->GetMethod(name);
		call->SetMethod(method);
	}
	ForwardTraversal::Visit(call);
}

void SymbolPass_Methods::Visit(Identifier *identifier)
{
	auto symbolTable = m_scopes.top();

	auto name = identifier->GetString();
	auto type = symbolTable->GetVariable(name);
	identifier->SetType(type);
	ForwardTraversal::Visit(identifier);
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

void SymbolTableDumper::Dump(Program *program)
{
	Utils::Logger::LogInfo("Global symbol table");
	program->Accept(*this);
}

void SymbolTableDumper::Visit(Program *program)
{
	Utils::Logger::LogInfo(program->GetSymbolTable()->ToString(), 1);
	ForwardTraversal::Visit(program);
}

void SymbolTableDumper::Visit(Module *module)
{
	Utils::Logger::LogInfo("Module symbol table: " + module->GetName());
	Utils::Logger::LogInfo(module->GetSymbolTable()->ToString(), 1);
	ForwardTraversal::Visit(module);
}

void SymbolTableDumper::Visit(Method *method)
{
	Utils::Logger::LogInfo("Method symbol table: " + method->GetName());
	Utils::Logger::LogInfo(method->GetSymbolTable()->ToString(), 1);
	ForwardTraversal::Visit(method);
}

}
