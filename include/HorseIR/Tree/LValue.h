#pragma once

#include <vector>

#include "HorseIR/Tree/Node.h"

#include "HorseIR/Semantics/SymbolTable/SymbolTable.h"
#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class LValue : virtual public Node
{
public:
	virtual LValue *Clone() const override = 0;

	// Properties

	virtual const Type *GetType() const = 0;
	virtual Type *GetType() = 0;

	// Symbol table

	const SymbolTable::Symbol *GetSymbol() const { return m_symbol; }
	void SetSymbol(const SymbolTable::Symbol *symbol) { m_symbol = symbol; }

protected:
	LValue() {}

	const SymbolTable::Symbol *m_symbol = nullptr;
};

}
