#include "Runtime/Interpreter.h"

#include "CUDA/Vector.h"

#include "HorseIR/Semantics/SemanticAnalysis.h"
#include "HorseIR/Utils/TypeUtils.h"

#include "Runtime/DataBuffers/FunctionBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"
#include "Runtime/DataBuffers/DataObjects/VectorData.h"

#include "Runtime/BuiltinExecutionEngine.h"
#include "Runtime/GPU/GPUExecutionEngine.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"

namespace Runtime {

std::vector<DataBuffer *> Interpreter::Execute(const HorseIR::Program *program)
{
	// Execute!

	auto entry = HorseIR::SemanticAnalysis::GetEntry(program);
	return Execute(entry, {});
}

std::vector<DataBuffer *> Interpreter::Execute(const HorseIR::FunctionDeclaration *function, const std::vector<DataBuffer *>& arguments)
{
	switch (function->GetKind())
	{
		case HorseIR::FunctionDeclaration::Kind::Definition:
		{
			return Execute(static_cast<const HorseIR::Function *>(function), arguments);
		}
		case HorseIR::FunctionDeclaration::Kind::Builtin:
		{
			return Execute(static_cast<const HorseIR::BuiltinFunction *>(function), arguments);
		}
		default:
		{
			Utils::Logger::LogError("Cannot execute function '" + function->GetName() + "'");
		}
	}
}

std::vector<DataBuffer *> Interpreter::Execute(const HorseIR::Function *function, const std::vector<DataBuffer *>& arguments)
{
	if (Utils::Options::Present(Utils::Options::Opt_Print_debug))
	{
		Utils::Logger::LogDebug("Executing function '" + function->GetName() + "'");
	}
	Utils::ScopedChrono chrono("Function '" + function->GetName() + "'");

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
	std::string name = "Builtin function";
	switch (function->GetPrimitive())
	{
		case HorseIR::BuiltinFunction::Primitive::GPUOrderLib:
		case HorseIR::BuiltinFunction::Primitive::GPUGroupLib:
		case HorseIR::BuiltinFunction::Primitive::GPUUniqueLib:
		case HorseIR::BuiltinFunction::Primitive::GPUJoinLib:
			name = "Library function";
	}

	Utils::ScopedChrono chrono(name + " '" + function->GetName() + "'");

	BuiltinExecutionEngine engine(m_runtime, m_program);
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

void Interpreter::Visit(const HorseIR::Statement *statement)
{
	Utils::Logger::LogError("Statement kind unimplemented");
}

void Interpreter::Visit(const HorseIR::DeclarationStatement *declarationS)
{
	// Ignore
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
		Utils::Logger::LogError("Invalid number of data objects for assignment. Expected " + std::to_string(assignS->GetTargetCount()) + ", received " + std::to_string(values.size()));
	}

	// Update the runtime data map, copying if necessary

	auto index = 0u;
	for (const auto& target : assignS->GetTargets())
	{
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
	Utils::Logger::LogError("Repeat statement unimpemented");
}

void Interpreter::Visit(const HorseIR::BlockStatement *blockS)
{
	Utils::Logger::LogError("Block statement unimpemented");
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
	Utils::Logger::LogError("Break statemenet unimpemented");
}

void Interpreter::Visit(const HorseIR::ContinueStatement *continueS)
{
	Utils::Logger::LogError("Continue statement unimpemented");
}

void Interpreter::Visit(const HorseIR::Expression *expression)
{
	Utils::Logger::LogError("Expression kind unimplemented");
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

	if (*castType != *dataType)
	{
		// Check the runtime casting success

		if (!HorseIR::TypeUtils::IsCastable(castType, dataType))
		{
			Utils::Logger::LogError("Invalid cast, cannot cast '" + HorseIR::PrettyPrinter::PrettyString(dataType) + "' to '" + HorseIR::PrettyPrinter::PrettyString(castType) + "'");
		}

		// Convert the data to the right type

		auto vectorBuffer = BufferUtils::GetBuffer<VectorBuffer>(data);
		auto vectorCastType = HorseIR::TypeUtils::GetType<HorseIR::BasicType>(castType);

		auto convertedBuffer = Cast(vectorCastType, vectorBuffer);
		m_environment.Insert(cast, {convertedBuffer});
	}
	else
	{
		m_environment.Insert(cast, {data});
	}
}

VectorBuffer *Interpreter::Cast(const HorseIR::BasicType *castType, VectorBuffer *input) const
{
	switch (input->GetType()->GetBasicKind())
	{
		case HorseIR::BasicType::BasicKind::Boolean:
		case HorseIR::BasicType::BasicKind::Char:
		case HorseIR::BasicType::BasicKind::Int8:
		{
			return Cast<std::int8_t>(castType, input);
		}
		case HorseIR::BasicType::BasicKind::Int16:
		{
			return Cast<std::int16_t>(castType, input);
		}
		case HorseIR::BasicType::BasicKind::Int32:
		{
			return Cast<std::int32_t>(castType, input);
		}
		case HorseIR::BasicType::BasicKind::Int64:
		{
			return Cast<std::int64_t>(castType, input);
		}
		case HorseIR::BasicType::BasicKind::Float32:
		{
			return Cast<float>(castType, input);
		}
		case HorseIR::BasicType::BasicKind::Float64:
		{
			return Cast<double>(castType, input);
		}
		case HorseIR::BasicType::BasicKind::String:
		case HorseIR::BasicType::BasicKind::Symbol:
		{
			return Cast<std::uint64_t>(castType, input);
		}
	}
	Utils::Logger::LogError("Invalid cast, cannot cast '" + HorseIR::PrettyPrinter::PrettyString(input->GetType()) + "' to '" + HorseIR::PrettyPrinter::PrettyString(castType) + "'");
}

template<class S>
VectorBuffer *Interpreter::Cast(const HorseIR::BasicType *castType, VectorBuffer *input) const
{
	const auto typedInput = BufferUtils::GetVectorBuffer<S>(input);
	switch (castType->GetBasicKind())
	{
		case HorseIR::BasicType::BasicKind::Boolean:
		case HorseIR::BasicType::BasicKind::Char:
		case HorseIR::BasicType::BasicKind::Int8:
		{
			return Cast<std::int8_t, S>(castType, typedInput);
		}
		case HorseIR::BasicType::BasicKind::Int16:
		{
			return Cast<std::int16_t, S>(castType, typedInput);
		}
		case HorseIR::BasicType::BasicKind::Int32:
		{
			return Cast<std::int32_t, S>(castType, typedInput);
		}
		case HorseIR::BasicType::BasicKind::Int64:
		{
			return Cast<std::int64_t, S>(castType, typedInput);
		}
		case HorseIR::BasicType::BasicKind::Float32:
		{
			return Cast<float, S>(castType, typedInput);
		}
		case HorseIR::BasicType::BasicKind::Float64:
		{
			return Cast<double, S>(castType, typedInput);
		}
		case HorseIR::BasicType::BasicKind::String:
		case HorseIR::BasicType::BasicKind::Symbol:
		{
			return Cast<std::uint64_t, S>(castType, typedInput);
		}
	}
	Utils::Logger::LogError("Invalid cast, cannot cast '" + HorseIR::PrettyPrinter::PrettyString(input->GetType()) + "' to '" + HorseIR::PrettyPrinter::PrettyString(castType) + "'");
}

template<class D, class S>
TypedVectorBuffer<D> *Interpreter::Cast(const HorseIR::BasicType *castType, TypedVectorBuffer<S> *input) const
{
	if constexpr(std::is_same<D, S>::value)
	{
		return input;
	}
	else
	{
		const auto& inputData = input->GetCPUReadBuffer()->GetValues();
		const auto size = inputData.size();

		CUDA::Vector<D> convertedData(size);

		for (auto i = 0u; i < size; ++i)
		{
			convertedData[i] = std::move(static_cast<D>(inputData[i]));
		}

		return new TypedVectorBuffer(new TypedVectorData<D>(castType, std::move(convertedData)));
	}
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

	auto function = call->GetFunctionLiteral()->GetFunction();
	m_environment.Insert(call, Execute(function, argumentsData));
}

void Interpreter::Visit(const HorseIR::Identifier *identifier)
{
	// Get the evaluated expression for the identifier

	m_environment.Insert(identifier, m_environment.Get(identifier->GetSymbol()));
}

void Interpreter::Visit(const HorseIR::FunctionLiteral *literal)
{
	// Store the function pointer in a buffer

	m_environment.Insert(literal, new FunctionBuffer(literal->GetFunction()));
}

template<typename T>
void Interpreter::VisitVectorLiteral(const HorseIR::TypedVectorLiteral<T> *literal)
{
	// Create a vector of values from the literal
	
	auto type = literal->GetType();
	if (!HorseIR::TypeUtils::IsType<HorseIR::BasicType>(type))
	{
		Utils::Logger::LogError("Invalid type '" + HorseIR::PrettyPrinter::PrettyString(type) + "' for vector literal");
	}
	auto basicType = HorseIR::TypeUtils::GetType<HorseIR::BasicType>(type);

	if constexpr(std::is_same<T, std::string>::value)
	{
		CUDA::Vector<std::uint64_t> vector;
		for (const auto& string : literal->GetValues())
		{
			vector.push_back(StringBucket::HashString(string));
		}
		m_environment.Insert(literal, new TypedVectorBuffer<std::uint64_t>(new TypedVectorData<std::uint64_t>(basicType, std::move(vector))));
	}
	else if constexpr(std::is_same<T, HorseIR::SymbolValue *>::value)
	{
		CUDA::Vector<std::uint64_t> vector;
		for (const auto& symbol : literal->GetValues())
		{
			vector.push_back(StringBucket::HashString(symbol->GetName()));
		}
		m_environment.Insert(literal, new TypedVectorBuffer<std::uint64_t>(new TypedVectorData<std::uint64_t>(basicType, std::move(vector))));
	}	
	else if constexpr(std::is_convertible<T, HorseIR::CalendarValue *>::value)
	{
		CUDA::Vector<std::int32_t> vector;
		for (const auto& value : literal->GetValues())
		{
			vector.push_back(value->GetEpochTime());
		}
		m_environment.Insert(literal, new TypedVectorBuffer<std::int32_t>(new TypedVectorData<std::int32_t>(basicType, std::move(vector))));
	}	
	else if constexpr(std::is_convertible<T, HorseIR::ExtendedCalendarValue *>::value)
	{
		CUDA::Vector<double> vector;
		for (const auto& value : literal->GetValues())
		{
			vector.push_back(value->GetExtendedEpochTime());
		}
		m_environment.Insert(literal, new TypedVectorBuffer<double>(new TypedVectorData<double>(basicType, std::move(vector))));
	}	
	else
	{
		auto& values = literal->GetValues();
		CUDA::Vector<T> vector(std::begin(values), std::end(values));
		m_environment.Insert(literal, new TypedVectorBuffer<T>(new TypedVectorData<T>(basicType, std::move(vector))));
	}
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
