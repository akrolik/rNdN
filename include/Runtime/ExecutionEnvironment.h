#pragma once

#include <stack>
#include <unordered_map>
#include <vector>

#include "HorseIR/Tree/Tree.h"

#include "Runtime/DataBuffers/DataBuffer.h"

namespace Runtime {

class ExecutionEnvironment
{
public:
	void Insert(const HorseIR::SymbolTable::Symbol *symbol, DataBuffer *buffer);
	void Insert(const HorseIR::Operand *operand, DataBuffer *buffer);
	void Insert(const HorseIR::Expression *expression, const std::vector<DataBuffer *>& buffers);

	DataBuffer *Get(const HorseIR::SymbolTable::Symbol *symbol) const;
	DataBuffer *Get(const HorseIR::Operand *operand) const;
	const std::vector<DataBuffer *>& Get(const HorseIR::Expression *expression) const;

	void PushStackFrame(const HorseIR::Function *function);
	std::vector<DataBuffer *> PopStackFrame();

	void InsertReturn(DataBuffer *buffer);

private:
	struct GlobalContext
	{
		std::unordered_map<const HorseIR::SymbolTable::Symbol *, DataBuffer *> variableMap;
		std::unordered_map<const HorseIR::Expression *, std::vector<DataBuffer *>> expressionMap;
	};

	std::unordered_map<const HorseIR::Module *, GlobalContext> m_globalContexts;

	struct FunctionContext
	{
		std::unordered_map<const HorseIR::SymbolTable::Symbol *, DataBuffer *> variableMap;
		std::unordered_map<const HorseIR::Expression *, std::vector<DataBuffer *>> expressionMap;
		std::vector<DataBuffer *> results;
	};

	std::stack<FunctionContext> m_functionContexts;
};

}
