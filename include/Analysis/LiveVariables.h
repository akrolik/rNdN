#pragma once

#include "HorseIR/Analysis/BackwardAnalysis.h"

#include "HorseIR/Semantics/SymbolTable/SymbolTable.h"
#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace Analysis {

struct LiveVariablesValue
{
	struct Hash {
		size_t operator() (const LiveVariablesValue* val) const
		{
			return std::hash<const HorseIR::SymbolTable::Symbol *>()(val->m_symbol);
		}
	};
	
 	struct Equals {
		bool operator() (const LiveVariablesValue *val1, const LiveVariablesValue *val2) const
		{
			return (*val1 == *val2);
		}
	};
 
	bool operator==(const LiveVariablesValue& other) const
	{
		return (m_symbol == other.m_symbol);
	}

	friend std::ostream& operator<<(std::ostream& os, const LiveVariablesValue& value);

	LiveVariablesValue(const HorseIR::SymbolTable::Symbol *symbol) : m_symbol(symbol) {}

	const HorseIR::SymbolTable::Symbol *GetSymbol() const { return m_symbol; }

private:
	const HorseIR::SymbolTable::Symbol *m_symbol = nullptr;
};

inline std::ostream& operator<<(std::ostream& os, const LiveVariablesValue& value)
{
	os << HorseIR::PrettyPrinter::PrettyString(value.m_symbol->node);
	return os;
}

class LiveVariables : public HorseIR::BackwardAnalysis<LiveVariablesValue>
{
public:
	using HorseIR::BackwardAnalysis<LiveVariablesValue>::BackwardAnalysis;

	void Visit(const HorseIR::VariableDeclaration *declaration) override;
	void Visit(const HorseIR::AssignStatement *assignS) override;
	void Visit(const HorseIR::Identifier *identifier) override;

	using SetType = HorseIR::BackwardAnalysis<LiveVariablesValue>::SetType;
	virtual SetType Merge(const SetType& s1, const SetType& s2) const override;

protected:
	void Kill(const HorseIR::SymbolTable::Symbol *symbol);

	bool m_isTarget = false;
};

}
