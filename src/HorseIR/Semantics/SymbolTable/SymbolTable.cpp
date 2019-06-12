#include "HorseIR/Semantics/SymbolTable/SymbolTable.h"

#include "HorseIR/Tree/Node.h"
#include "HorseIR/Tree/Module.h"
#include "HorseIR/Tree/FunctionDeclaration.h"
#include "HorseIR/Tree/VariableDeclaration.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"

namespace HorseIR {

std::ostream& operator<<(std::ostream& os, const SymbolTable::Symbol& value)
{
	switch (value.kind)
	{
		case SymbolTable::Symbol::Kind::Module:
			os << "[module]";
			break;
		case SymbolTable::Symbol::Kind::Function:
			os << "[function] = " << PrettyPrinter::PrettyString(value.node, true);
			break;
		case SymbolTable::Symbol::Kind::Variable:
			os << "[variable] = " << PrettyPrinter::PrettyString(value.node);
			break;
	}
	return os;
}

bool SymbolTable::ContainsSymbol(const std::string& name) const
{
	return m_table.find(name) != m_table.end();
}

bool SymbolTable::ContainsSymbol(const Symbol *symbol) const
{
	for (const auto val : m_table)
	{
		if (val.second == symbol)
		{
			return true;
		}
	}
	return false;
}

SymbolTable::Symbol *SymbolTable::GetSymbol(const std::string& name) const
{
	// Get symbol from the table, recursively traversing to parent

	if (ContainsSymbol(name))
	{
		return m_table.at(name);
	}
	if (m_importTable != nullptr && m_importTable->ContainsSymbol(name))
	{
		return m_importTable->GetSymbol(name);
	}
	if (m_parent != nullptr)
	{
		return m_parent->GetSymbol(name);
	}
	Utils::Logger::LogError("'" + name + "' is not defined in current program");
	return nullptr;
}

const Module *SymbolTable::GetModule(const std::string& name) const
{
	auto symbol = GetSymbol(name);
	if (symbol == nullptr)
	{
		Utils::Logger::LogError("Module " + name + " is not defined in current program");
	}
	else if (symbol->kind != SymbolTable::Symbol::Kind::Module)
	{
		Utils::Logger::LogError("'" + name + "' is not a module");
	}
	return static_cast<const Module *>(symbol->node);
}

const FunctionDeclaration *SymbolTable::GetFunction(const std::string& name) const
{
	auto symbol = GetSymbol(name);
	if (symbol == nullptr)
	{
		Utils::Logger::LogError("Function '" + name + "' cannot be found in the current module scope");
	}
	else if (symbol->kind != SymbolTable::Symbol::Kind::Function)
	{
		Utils::Logger::LogError("'" + name + "' is not a function");
	}
	return static_cast<const FunctionDeclaration *>(symbol->node);
}

const VariableDeclaration *SymbolTable::GetVariable(const std::string& name) const
{
	auto symbol = GetSymbol(name);
	if (symbol == nullptr)
	{
		Utils::Logger::LogError("Variable '" + name + "' cannot be found in the current function scope");
	}
	else if (symbol->kind != SymbolTable::Symbol::Kind::Variable)
	{
		Utils::Logger::LogError("'" + name + "' is not a variable");
	}
	
	// This uses a dynamic cast to handle virtual inheritance with LValue
	return dynamic_cast<const VariableDeclaration *>(symbol->node);
}

void SymbolTable::AddSymbol(const std::string& name, Symbol *symbol, bool replace)
{
	// Check if symbol is already defined at this scope, and if we are allowing replacement

	if (m_table.find(name) != m_table.end())
	{
		if (replace)
		{
			// Replace the value in the map, this should only be used for imports
			m_table[name] = symbol;
		}
		else
		{
			Utils::Logger::LogError("Identifier " + name + " is already defined in current scope");
		}
	}
	else
	{
		m_table.insert({name, symbol});
	}
}

}
