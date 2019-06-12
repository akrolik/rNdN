#pragma once

#include <vector>

#include "HorseIR/Tree/Node.h"

#include "HorseIR/Semantics/SymbolTable/SymbolTable.h"
#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class LValue : virtual public Node
{
public:
	virtual const std::vector<Type *> GetTypes() const = 0;

	SymbolTable::Symbol *GetSymbol() const { return m_symbol; }
	void SetSymbol(SymbolTable::Symbol *symbol) { m_symbol = symbol; }

protected:
	LValue() {}

	SymbolTable::Symbol *m_symbol = nullptr;
};

}
