#include "HorseIR/SymbolTable.h"

namespace HorseIR {

void SymbolTable::Build(HorseIR::Method *method)
{
	m_table.clear();
	method->Accept(*this);
}

Type *SymbolTable::GetType(const std::string& identifier)
{
	return m_table.at(identifier);
}

void SymbolTable::Visit(Parameter *parameter)
{
	m_table.insert({parameter->GetName(), parameter->GetType()});
	ForwardTraversal::Visit(parameter);
}

void SymbolTable::Visit(AssignStatement *assign)
{
	m_table.insert({assign->GetTargetName(), assign->GetType()});
	ForwardTraversal::Visit(assign);
}

void SymbolTable::Visit(Identifier *identifier)
{
	identifier->SetType(GetType(identifier->GetString()));
	ForwardTraversal::Visit(identifier);
}

}
