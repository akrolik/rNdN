#include "HorseIR/SymbolTable.h"

namespace HorseIR {

void SymbolTable::Build(HorseIR::Method *method)
{
	m_table.clear();
	method->Accept(*this);
}

HorseIR::Type *SymbolTable::GetType(const std::string& identifier)
{
	return m_table.at(identifier);
}

void SymbolTable::Visit(Parameter *parameter)
{
	m_table.insert({parameter->GetName(), parameter->GetType()});

}

void SymbolTable::Visit(AssignStatement *assign)
{
	m_table.insert({assign->GetIdentifier(), assign->GetType()});
}

}
