#pragma once

#include <string>
#include <unordered_set>

#include "HorseIR/Analysis/ForwardAnalysis.h"

#include "HorseIR/Semantics/SymbolTable/SymbolTable.h"
#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace Analysis {

struct ReachingDefinitionsValue
{
	struct Hash {
		size_t operator() (const ReachingDefinitionsValue* val) const
		{
			return std::hash<const HorseIR::SymbolTable::Symbol *>()(val->m_symbol);
		}
	};
	
 	struct Equals {
		bool operator() (const ReachingDefinitionsValue *val1, const ReachingDefinitionsValue *val2) const
		{
			return (*val1 == *val2);
		}
	};
 
	bool operator==(const ReachingDefinitionsValue& other) const
	{
		return (m_symbol == other.m_symbol && m_statements == other.m_statements);
	}

	friend std::ostream& operator<<(std::ostream& os, const ReachingDefinitionsValue& value);

	ReachingDefinitionsValue(const HorseIR::SymbolTable::Symbol *symbol, const HorseIR::AssignStatement *statement) : m_symbol(symbol), m_statements({statement}) {}
	ReachingDefinitionsValue(const HorseIR::SymbolTable::Symbol *symbol, const std::unordered_set<const HorseIR::AssignStatement *>& statements) : m_symbol(symbol), m_statements(statements) {}

	const HorseIR::SymbolTable::Symbol *GetSymbol() const { return m_symbol; }
	const std::unordered_set<const HorseIR::AssignStatement *>& GetStatements() const { return m_statements; }

	void AddStatement(const HorseIR::AssignStatement *statement) { m_statements.insert(statement); }
	void AddStatements(const std::unordered_set<const HorseIR::AssignStatement *>& statements) { m_statements.insert(statements.begin(), statements.end()); }

private:
	const HorseIR::SymbolTable::Symbol *m_symbol = nullptr;
	std::unordered_set<const HorseIR::AssignStatement *> m_statements;
};

inline std::ostream& operator<<(std::ostream& os, const ReachingDefinitionsValue& value)
{
	os << HorseIR::PrettyPrinter::PrettyString(value.m_symbol->node) << "->[";

	bool first = true;
	for (const auto& statement : value.m_statements)
	{
		if (!first)
		{
			os << ", ";
		}
		first = false;
		os << HorseIR::PrettyPrinter::PrettyString(statement->GetExpression());
	}
	os << "]";
	return os;
}

class ReachingDefinitions : public HorseIR::ForwardAnalysis<ReachingDefinitionsValue>
{
public:
	using HorseIR::ForwardAnalysis<ReachingDefinitionsValue>::ForwardAnalysis;

	void Visit(const HorseIR::AssignStatement *assignS) override;
	void Visit(const HorseIR::BlockStatement *blockS) override;

	using SetType = HorseIR::ForwardAnalysis<ReachingDefinitionsValue>::SetType;
	virtual SetType Merge(const SetType& s1, const SetType& s2) const override;
};

}
