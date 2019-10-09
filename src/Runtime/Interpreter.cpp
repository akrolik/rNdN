#include "Runtime/Interpreter.h"

#include "HorseIR/Utils/TypeUtils.h"

#include "Runtime/DataBuffers/VectorBuffer.h"
#include "Runtime/DataBuffers/DataObjects/VectorData.h"

#include "Runtime/BuiltinExecutionEngine.h"
#include "Runtime/GPUExecutionEngine.h"

#include "Utils/Logger.h"

namespace Runtime {

std::vector<DataBuffer *> Interpreter::Execute(const HorseIR::FunctionDeclaration *function, const std::vector<DataBuffer *>& arguments)
{
	switch (function->GetKind())
	{
		case HorseIR::FunctionDeclaration::Kind::Definition:
			return Execute(static_cast<const HorseIR::Function *>(function), arguments);
		case HorseIR::FunctionDeclaration::Kind::Builtin:
			return Execute(static_cast<const HorseIR::BuiltinFunction *>(function), arguments);
	}

	Utils::Logger::LogError("Cannot execute function '" + function->GetName() + "'");
}

std::vector<DataBuffer *> Interpreter::Execute(const HorseIR::Function *function, const std::vector<DataBuffer *>& arguments)
{
	Utils::Logger::LogInfo("Executing function '" + function->GetName() + "'");

	if (function->IsKernel())
	{
		// Pass function execution to the GPU engine

		GPUExecutionEngine engine(m_runtime, m_program);
		return engine.Execute(function, arguments);
	}
	else
	{
		// Push onto the function stack, and add all parameters to the context

		m_environment.PushStackFrame(function);

		auto i = 0u;
		for (const auto& parameter : function->GetParameters())
		{
			m_environment.Insert(parameter->GetSymbol(), arguments.at(i++));
		}

		// Execute the function body
		
		for (const auto& statement : function->GetStatements())
		{
			statement->Accept(*this);
		}

		// Gather the return objects and pop the function stack

		return m_environment.PopStackFrame();
	}
}

std::vector<DataBuffer *> Interpreter::Execute(const HorseIR::BuiltinFunction *function, const std::vector<DataBuffer *>& arguments)
{
	BuiltinExecutionEngine engine(m_runtime);
	return engine.Execute(function, arguments);
}

void Interpreter::InitializeModule(const HorseIR::Module *module)
{
	//GLOBAL: Handle modules. We should transitively initialize modules by import. Cyclical imports are problematic
}

void Interpreter::Visit(const HorseIR::GlobalDeclaration *global)
{
	Utils::Logger::LogError("Unimplemented");
}

void Interpreter::Visit(const HorseIR::AssignStatement *assignS)
{
	// Evaluate the RHS of the assignment and update the context map

	auto expression = assignS->GetExpression();
	expression->Accept(*this);


	// Check the expression return count is the same as the number of targets

	auto& values = m_environment.Get(expression);
	if (values.size() != assignS->GetTargetCount())
	{
		Utils::Logger::LogError("Invalid number of data objects for assignment. Expected " + std::to_string(assignS->GetTargetCount()) + ", received " + std::to_string(values.size()) + "'");
	}

	// Update the runtime data map, copying if necessary

	auto index = 0u;
	for (const auto& target : assignS->GetTargets())
	{
		//TODO: Copy if this is a plain assignment. We assume that functions return a new variable
		m_environment.Insert(target->GetSymbol(), values.at(index++));
	}
}

void Interpreter::Visit(const HorseIR::ExpressionStatement *expressionS)
{
	// Evaluate the expression

	auto expression = expressionS->GetExpression();
	expression->Accept(*this);
}

void Interpreter::Visit(const HorseIR::IfStatement *ifS)
{
	Utils::Logger::LogError("Unimpemented");
}

void Interpreter::Visit(const HorseIR::WhileStatement *whileS)
{
	Utils::Logger::LogError("Unimpemented");
}

void Interpreter::Visit(const HorseIR::RepeatStatement *repeatS)
{
	Utils::Logger::LogError("Unimpemented");
}

void Interpreter::Visit(const HorseIR::BlockStatement *blockS)
{
	Utils::Logger::LogError("Unimpemented");
}

void Interpreter::Visit(const HorseIR::ReturnStatement *returnS)
{
	// Get the data item associated with each operand

	for (const auto& operand : returnS->GetOperands())
	{
		operand->Accept(*this);
		m_environment.InsertReturn(m_environment.Get(operand));
	}
}

void Interpreter::Visit(const HorseIR::BreakStatement *breakS)
{
	Utils::Logger::LogError("Unimpemented");
}

void Interpreter::Visit(const HorseIR::ContinueStatement *continueS)
{
	Utils::Logger::LogError("Unimpemented");
}

void Interpreter::Visit(const HorseIR::CastExpression *cast)
{
	// Evaluate the cast expression

	auto expression = cast->GetExpression();
	expression->Accept(*this);

	// Gather all the types necessary for casting

	auto& dataObjects = m_environment.Get(expression);
	if (dataObjects.size() != 1)
	{
		Utils::Logger::LogError("Invalid cast, expression may only have a single value");
	}

	auto data = dataObjects.at(0);

	auto dataType = data->GetType();
	auto castType = cast->GetCastType();

	// Check the runtime casting success

	if (!HorseIR::TypeUtils::IsCastable(castType, dataType))
	{
		Utils::Logger::LogError("Invalid cast, cannot cast '" + HorseIR::PrettyPrinter::PrettyString(dataType) + "' to '" + HorseIR::PrettyPrinter::PrettyString(castType) + "'");
	}

	m_environment.Insert(cast, {data});
}

void Interpreter::Visit(const HorseIR::CallExpression *call)
{
	// Evaluate arguments and collect the data objects

	std::vector<DataBuffer *> argumentsData;
	for (const auto& argument : call->GetArguments())
	{
		argument->Accept(*this);
		argumentsData.push_back(m_environment.Get(argument));
	}

	// Execute function and store result for the function invocation

	auto result = Execute(call->GetFunctionLiteral()->GetFunction(), argumentsData);
	m_environment.Insert(call, result);
}

void Interpreter::Visit(const HorseIR::Identifier *identifier)
{
	// Get the evaluated expression for the identifier

	m_environment.Insert(identifier, m_environment.Get(identifier->GetSymbol()));
}

template<typename T>
void Interpreter::VisitVectorLiteral(const HorseIR::TypedVectorLiteral<T> *literal)
{
	// Create a vector of values from the literal
	
	auto& types = literal->GetTypes();
	if (!HorseIR::TypeUtils::IsSingleType(types))
	{
		Utils::Logger::LogError("Vector literal expected a single type");
	}

	auto type = HorseIR::TypeUtils::GetSingleType(types);
	if (!HorseIR::TypeUtils::IsType<HorseIR::BasicType>(type))
	{
		Utils::Logger::LogError("Invalid type '" + HorseIR::PrettyPrinter::PrettyString(type) + "' for vector literal");
	}

	auto basicType = HorseIR::TypeUtils::GetType<HorseIR::BasicType>(type);
	auto vector = new TypedVectorData<T>(basicType, literal->GetValues());
	m_environment.Insert(literal, new TypedVectorBuffer<T>(vector));
}

void Interpreter::Visit(const HorseIR::BooleanLiteral *literal)
{
	VisitVectorLiteral(literal);
}

void Interpreter::Visit(const HorseIR::CharLiteral *literal)
{
	VisitVectorLiteral(literal);
}

void Interpreter::Visit(const HorseIR::Int8Literal *literal)
{
	VisitVectorLiteral(literal);
}

void Interpreter::Visit(const HorseIR::Int16Literal *literal)
{
	VisitVectorLiteral(literal);
}

void Interpreter::Visit(const HorseIR::Int32Literal *literal)
{
	VisitVectorLiteral(literal);
}

void Interpreter::Visit(const HorseIR::Int64Literal *literal)
{
	VisitVectorLiteral(literal);
}

void Interpreter::Visit(const HorseIR::Float32Literal *literal)
{
	VisitVectorLiteral(literal);
}

void Interpreter::Visit(const HorseIR::Float64Literal *literal)
{
	VisitVectorLiteral(literal);
}

void Interpreter::Visit(const HorseIR::ComplexLiteral *literal)
{
	VisitVectorLiteral(literal);
}

void Interpreter::Visit(const HorseIR::StringLiteral *literal)
{
	VisitVectorLiteral(literal);
}

void Interpreter::Visit(const HorseIR::SymbolLiteral *literal)
{
	VisitVectorLiteral(literal);
}

void Interpreter::Visit(const HorseIR::DatetimeLiteral *literal)
{
	VisitVectorLiteral(literal);
}

void Interpreter::Visit(const HorseIR::MonthLiteral *literal)
{
	VisitVectorLiteral(literal);
}

void Interpreter::Visit(const HorseIR::DateLiteral *literal)
{
	VisitVectorLiteral(literal);
}

void Interpreter::Visit(const HorseIR::MinuteLiteral *literal)
{
	VisitVectorLiteral(literal);
}

void Interpreter::Visit(const HorseIR::SecondLiteral *literal)
{
	VisitVectorLiteral(literal);
}

void Interpreter::Visit(const HorseIR::TimeLiteral *literal)
{
	VisitVectorLiteral(literal);
}

}