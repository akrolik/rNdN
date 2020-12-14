#include "Runtime/ExecutionEnvironment.h"

namespace Runtime {

//GLOBAL: Add global variable support to all symbol functions. Have a link from symbol to module, and from function to module

void ExecutionEnvironment::Insert(const HorseIR::SymbolTable::Symbol *symbol, DataBuffer *object)
{
	auto& context = m_functionContexts.top();
	context.variableMap[symbol] = object;
}

void ExecutionEnvironment::Insert(const HorseIR::Operand *operand, DataBuffer *object)
{
	auto& context = m_functionContexts.top();
	context.expressionMap[operand] = {object};
}

void ExecutionEnvironment::Insert(const HorseIR::Expression *expression, const std::vector<DataBuffer *>& objects)
{
	auto& context = m_functionContexts.top();
	context.expressionMap[expression] = objects;
}

const DataBuffer *ExecutionEnvironment::Get(const HorseIR::SymbolTable::Symbol *symbol) const
{
	auto& context = m_functionContexts.top();
	return context.variableMap.at(symbol);
}

DataBuffer *ExecutionEnvironment::Get(const HorseIR::SymbolTable::Symbol *symbol)
{
	auto& context = m_functionContexts.top();
	return context.variableMap.at(symbol);
}

const DataBuffer *ExecutionEnvironment::Get(const HorseIR::Operand *operand) const
{
	auto& context = m_functionContexts.top();
	return context.expressionMap.at(operand).at(0);
}

DataBuffer *ExecutionEnvironment::Get(const HorseIR::Operand *operand)
{
	auto& context = m_functionContexts.top();
	return context.expressionMap.at(operand).at(0);
}

std::vector<const DataBuffer *> ExecutionEnvironment::Get(const HorseIR::Expression *expression) const
{
	auto& context = m_functionContexts.top();
	auto& expressions = context.expressionMap.at(expression);
	return { std::begin(expressions), std::end(expressions) };
}

std::vector<DataBuffer *>& ExecutionEnvironment::Get(const HorseIR::Expression *expression)
{
	auto& context = m_functionContexts.top();
	return context.expressionMap.at(expression);
}

void ExecutionEnvironment::PushStackFrame(const HorseIR::Function *function)
{
	m_functionContexts.emplace();
}

std::vector<DataBuffer *> ExecutionEnvironment::PopStackFrame()
{
	auto results = m_functionContexts.top().results;
	m_functionContexts.pop();
	return results;
}

void ExecutionEnvironment::InsertReturn(DataBuffer *object)
{
	auto& context = m_functionContexts.top();
	context.results.push_back(object);
}

}
