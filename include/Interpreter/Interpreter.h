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
	void Execute(HorseIR::Method *method);
	void Execute(HorseIR::BuiltinMethod *method);

	void Visit(HorseIR::AssignStatement *assign) override;
	void Visit(HorseIR::CastExpression *cast) override;
	void Visit(HorseIR::CallExpression *call) override;

private:
	Runtime::Runtime& m_runtime;

	//TODO:
	HorseIR::Program *m_program = nullptr;

	std::unordered_map<std::string, Runtime::DataObject *> m_dataMap;
	std::unordered_map<HorseIR::Expression *, Runtime::DataObject *> m_expressionMap;
};

}
