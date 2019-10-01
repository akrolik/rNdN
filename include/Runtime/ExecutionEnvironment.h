#pragma once

#include <stack>
#include <unordered_map>
#include <vector>

#include "HorseIR/Tree/Tree.h"

#include "Runtime/DataObjects/DataObject.h"

namespace Runtime {

class ExecutionEnvironment
{
public:
	void Insert(const HorseIR::SymbolTable::Symbol *symbol, DataObject *object);
	void Insert(const HorseIR::Operand *operand, DataObject *object);
	void Insert(const HorseIR::Expression *expression, const std::vector<DataObject *>& objects);

	DataObject *Get(const HorseIR::SymbolTable::Symbol *symbol) const;
	DataObject *Get(const HorseIR::Operand *operand) const;
	const std::vector<DataObject *>& Get(const HorseIR::Expression *expression) const;

	void PushStackFrame(const HorseIR::Function *function);
	std::vector<DataObject *> PopStackFrame();

	void InsertReturn(DataObject *object);

private:
	struct GlobalContext
	{
		std::unordered_map<const HorseIR::SymbolTable::Symbol *, DataObject *> variableMap;
		std::unordered_map<const HorseIR::Expression *, std::vector<DataObject *>> expressionMap;
	};

	std::unordered_map<const HorseIR::Module *, GlobalContext> m_globalContexts;

	struct FunctionContext
	{
		std::unordered_map<const HorseIR::SymbolTable::Symbol *, DataObject *> variableMap;
		std::unordered_map<const HorseIR::Expression *, std::vector<DataObject *>> expressionMap;
		std::vector<DataObject *> results;
	};

	std::stack<FunctionContext> m_functionContexts;
};

}
