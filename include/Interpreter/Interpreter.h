#pragma once

#include "HorseIR/Traversal/Visitor.h"

#include "Runtime/DataObject.h"
#include "Runtime/Runtime.h"

namespace Interpreter {

class Interpreter : public HorseIR::Visitor
{
public:
	Interpreter(Runtime::Runtime& runtime) : m_runtime(runtime) {}

	void Execute(HorseIR::Program *program);
	Runtime::DataObject *Execute(const HorseIR::Method *method, const std::vector<HorseIR::Expression *>& arguments);
	Runtime::DataObject *Execute(const HorseIR::BuiltinMethod *method, const std::vector<HorseIR::Expression *>& arguments);

	void Visit(HorseIR::AssignStatement *assign) override;
	void Visit(HorseIR::ReturnStatement *ret) override;
	void Visit(HorseIR::CastExpression *cast) override;
	void Visit(HorseIR::CallExpression *call) override;
	void Visit(HorseIR::Identifier *identifier) override;
	void Visit(HorseIR::Symbol *symbol) override;

private:
	Runtime::Runtime& m_runtime;

	//TODO:
	HorseIR::Program *m_program = nullptr;
	Runtime::DataObject *m_result = nullptr;

	std::unordered_map<std::string, Runtime::DataObject *> m_variableMap;
	std::unordered_map<HorseIR::Expression *, Runtime::DataObject *> m_expressionMap;
};

}
