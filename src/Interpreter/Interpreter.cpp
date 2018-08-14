#include "Interpreter/Interpreter.h"

#include "CUDA/Buffer.h"
#include "CUDA/Kernel.h"
#include "CUDA/KernelInvocation.h"
#include "CUDA/Module.h"

#include "HorseIR/TypeUtils.h"
#include "HorseIR/Analysis/EntryAnalysis.h"
#include "HorseIR/Analysis/ShapeAnalysis.h"
#include "HorseIR/Tree/BuiltinMethod.h"
#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"
#include "HorseIR/Tree/Expressions/CastExpression.h"
#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/Literals/SymbolLiteral.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"
#include "HorseIR/Tree/Statements/ReturnStatement.h"

#include "PTX/Program.h"

#include "Runtime/JITCompiler.h"
#include "Runtime/DataObjects/DataList.h"
#include "Runtime/DataObjects/DataVector.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Interpreter {

void Interpreter::Execute(HorseIR::Program *program)
{
	Utils::Logger::LogSection("Starting program execution");

	HorseIR::EntryAnalysis entryAnalysis;
	entryAnalysis.Analyze(program);
	auto result = Execute(entryAnalysis.GetEntry(), {});
	result->Dump();
}

Runtime::DataObject *Interpreter::Execute(HorseIR::MethodDeclaration *method, const std::vector<HorseIR::Expression *>& arguments)
{
	switch (method->GetKind())
	{
		case HorseIR::MethodDeclaration::Kind::Definition:
			return Execute(static_cast<HorseIR::Method *>(method), arguments);
		case HorseIR::MethodDeclaration::Kind::Builtin:
			return Execute(static_cast<HorseIR::BuiltinMethod *>(method), arguments);
	}

	Utils::Logger::LogError("Cannot execute method '" + method->GetName() + "'");
}

Runtime::DataObject *Interpreter::Execute(HorseIR::Method *method, const std::vector<HorseIR::Expression *>& arguments)
{
	Utils::Logger::LogInfo("Executing method '" + method->GetName() + "'");

	if (method->IsKernel())
	{
		// Run the shape and type analyses

		HorseIR::ShapeAnalysis shapeAnalysis;
		unsigned int i = 0;
		for (auto& parameter : method->GetParameters())
		{
			auto type = parameter->GetType();
			auto argument = arguments.at(i);
			switch (type->GetKind())
			{
				case HorseIR::Type::Kind::Basic:
				{
					auto argumentData = static_cast<Runtime::DataVector *>(m_expressionMap.at(argument));
					// parameter->SetShape(new HorseIR::Shape(HorseIR::Shape::Kind::Vector, argumentData->GetElementCount()));
					break;
				}
				default:
					Utils::Logger::LogError("Unsupported argument type " + type->ToString());
			}
			++i;
		}
		shapeAnalysis.Analyze(method);

		// Compile the HorseIR to PTX code using the current device

		auto& gpu = m_runtime.GetGPUManager();
		auto& device = gpu.GetCurrentDevice();

		Codegen::TargetOptions targetOptions;
		targetOptions.ComputeCapability = device->GetComputeCapability();
		targetOptions.MaxBlockSize = device->GetMaxThreadsDimension(0);
		targetOptions.WarpSize = device->GetWarpSize();

		//TODO: Get the input geometry size from the table
		Codegen::InputOptions inputOptions;
		inputOptions.InputSize = 2048;

		Runtime::JITCompiler compiler(targetOptions, inputOptions);
		PTX::Program *ptxProgram = compiler.Compile({method});

		// Optimize the generated PTX program

		if (Utils::Options::Get<>(Utils::Options::Opt_Optimize))
		{
			compiler.Optimize(ptxProgram);
		}

		// Create and load the CUDA module for the program

		CUDA::Module cModule = gpu.AssembleProgram(ptxProgram);

		// Compute the number of input arguments: method arguments + return + (dynamic size)

		unsigned int kernelArgumentCount = arguments.size() + 1;
		if (inputOptions.InputSize == Codegen::InputOptions::DynamicSize)
		{
			kernelArgumentCount++;
		}

                // Fetch the handle to the GPU entry function

		CUDA::Kernel kernel(method->GetName(), kernelArgumentCount, cModule);
		auto& kernelOptions = ptxProgram->GetEntryFunction(method->GetName())->GetOptions();

		Utils::Logger::LogInfo("Generated program for method '" + method->GetName() + "' with options");
		Utils::Logger::LogInfo(kernelOptions.ToString(), 1);

		Utils::Logger::LogSection("Continuing program execution");

		auto timeExec_start = Utils::Chrono::Start();

		// Initialize kernel invocation

		CUDA::KernelInvocation invocation(kernel);
                                
		// Compute the number of threads based on the kernel configuration. The default is to
		// use all threads available in a compute unit

		auto threadCount = targetOptions.MaxBlockSize;
		auto kernelThreadCount = kernelOptions.GetThreadCount();
		if (kernelThreadCount == PTX::Function::Options::DynamicThreadCount && kernelOptions.GetThreadMultiple() != 0)
		{
			// Maximize the thread count based on the GPU and thread multiple

			auto multiple = kernelOptions.GetThreadMultiple();
			threadCount = multiple * (threadCount / multiple);
		}
		else
		{
			// Fixed number of threads

			if (kernelThreadCount > targetOptions.MaxBlockSize)
			{
				Utils::Logger::LogError("Thread count " + std::to_string(kernelThreadCount) + " set by kernel " + kernel.GetName() + " is not supported by target (MaxBlockSize= " + std::to_string(targetOptions.MaxBlockSize));
			}
			threadCount = kernelThreadCount;
		}

		invocation.SetBlockShape(threadCount, 1, 1);
		invocation.SetGridShape((inputOptions.InputSize - 1) / threadCount + 1, 1, 1);

		unsigned int index = 0;
		for (const auto& argument : arguments)
		{
			auto argumentType = argument->GetType();
			Runtime::DataVector *argumentData = nullptr;

			switch (argumentType->GetKind())
			{
				case HorseIR::Type::Kind::Basic:
					argumentData = static_cast<Runtime::DataVector *>(m_expressionMap.at(argument));
					break;
				default:
					Utils::Logger::LogError("Unsupported argument type " + argumentType->ToString());
			}

			Utils::Logger::LogInfo("Transferring input argument '" + argument->ToString() + "' [" + argumentType->ToString() + "(" + std::to_string(argumentData->GetElementSize()) + " bytes) x " + std::to_string(argumentData->GetElementCount()) + "]");

			//TODO: All buffers should be padded to a multiple of the thread count
			//TODO: Build a GPU buffer manager
			auto buffer = new CUDA::Buffer(argumentData->GetData(), argumentData->GetDataSize());

			buffer->AllocateOnGPU();
			buffer->TransferToGPU();

			invocation.SetParameter(index++, *buffer);
		}

		if (inputOptions.InputSize == Codegen::InputOptions::DynamicSize)
		{
			CUDA::TypedConstant<uint64_t> sizeConstant(inputOptions.InputSize);
			invocation.SetParameter(index++, sizeConstant);
		}

		auto returnType = method->GetReturnType();
		Runtime::DataVector *returnData = nullptr;

		switch (returnType->GetKind())
		{
			case HorseIR::Type::Kind::Basic:
				//TODO: Use shape information to allocate the correct size
				returnData = Runtime::DataVector::CreateVector(static_cast<HorseIR::BasicType *>(returnType), 1);
				break;
			defualt:
				Utils::Logger::LogError("Unsupported return type " + returnType->ToString());
		}

		Utils::Logger::LogInfo("Initializing return argument [" + returnType->ToString() + "(" + std::to_string(returnData->GetElementSize()) + " bytes) x " + std::to_string(returnData->GetElementCount()) + "]");

		CUDA::Buffer returnBuffer(returnData->GetData(), returnData->GetDataSize());
		returnBuffer.AllocateOnGPU();
		returnBuffer.TransferToGPU();
		invocation.SetParameter(index++, returnBuffer);
		
		// Configure the dynamic shared memory according to the kernel

		invocation.SetSharedMemorySize(kernelOptions.GetSharedMemorySize());

		// Launch the kernel!

		invocation.Launch();

		// Complete the execution by transferring the results back to the host

		returnBuffer.TransferToCPU();

		auto timeExec = Utils::Chrono::End(timeExec_start);

		Utils::Logger::LogTiming("Kernel execution", timeExec);

		return returnData;
	}
	else
	{
		if (method->GetParameters().size() > 0)
		{
			Utils::Logger::LogError("Cannot interpret method with input parameters");
		}
		
		for (const auto& statement : method->GetStatements())
		{
			statement->Accept(*this);
		}
		return m_result;
	}
}

Runtime::DataObject *Interpreter::Execute(HorseIR::BuiltinMethod *method, const std::vector<HorseIR::Expression *>& arguments)
{
	Utils::Logger::LogInfo("Executing builtin method '" + method->GetName() + "'");

	switch (method->GetKind())
	{
		case HorseIR::BuiltinMethod::Kind::Enlist:
		{
			auto vector = static_cast<Runtime::DataVector *>(m_expressionMap.at(arguments.at(0)));
			auto list = new Runtime::DataList(vector->GetType(), vector);

			return list;
		}
		case HorseIR::BuiltinMethod::Kind::Table:
		{
			auto columnNames = static_cast<Runtime::TypedDataVector<std::string> *>(m_expressionMap.at(arguments.at(0)));
			auto columnValues = static_cast<Runtime::DataList *>(m_expressionMap.at(arguments.at(1)));

			auto table = new Runtime::DataTable(1);

			unsigned int i = 0;
			for (const auto& columnName : columnNames->GetValues())
			{
				table->AddColumn(columnName, columnValues->GetElement(i++));
			}

			return table;
		}
		case HorseIR::BuiltinMethod::Kind::ColumnValue:
		{
			auto table = static_cast<Runtime::DataTable *>(m_expressionMap.at(arguments.at(0)));
			auto columnSymbol = static_cast<const HorseIR::SymbolLiteral *>(arguments.at(1));

			if (columnSymbol->GetCount() != 1)
			{
				Utils::Logger::LogError("Builtin function '" + method->GetName() + "' expects a single column argument");
			}

			return table->GetColumn(columnSymbol->GetValue(0));
		}
		case HorseIR::BuiltinMethod::Kind::LoadTable:
		{
			auto dataRegistry = m_runtime.GetDataRegistry();
			auto tableSymbol = static_cast<const HorseIR::SymbolLiteral *>(arguments.at(0));

			if (tableSymbol->GetCount() != 1)
			{
				Utils::Logger::LogError("Builtin function '" + method->GetName() + "' expects a single table argument");
			}

			return dataRegistry.GetTable(tableSymbol->GetValue(0));
		}
		default:
			Utils::Logger::LogError("Builtin function '" + method->GetName() + "' not implemented");
	}
}

void Interpreter::Visit(HorseIR::AssignStatement *assign)
{
	// Evaluate the RHS of the assignment

	auto expression = assign->GetExpression();
	expression->Accept(*this);

	// Update the runtime data map

	auto declaration = assign->GetDeclaration();

	m_variableMap.insert({declaration->GetName(), m_expressionMap.at(expression)});
}

void Interpreter::Visit(HorseIR::ReturnStatement *ret)
{
	m_result = m_variableMap.at(ret->GetIdentifier()->GetString());
}

void Interpreter::Visit(HorseIR::CastExpression *cast)
{
	// Evaluate the cast expression

	auto expression = cast->GetExpression();
	expression->Accept(*this);

	// Gather all the types necessary for casting

	auto expressionType = expression->GetType();

	auto data = m_expressionMap.at(expression);
	auto dataType = data->GetType();

	auto castType = cast->GetCastType();

	// If the expression has no type, then this is a runtime cast check

	if (expressionType == nullptr && *dataType != *castType)
	{
		Utils::Logger::LogError("Invalid cast, " + dataType->ToString() + " cannot be cast to " + castType->ToString());
	}

	m_expressionMap.insert({cast, data});
}

void Interpreter::Visit(HorseIR::CallExpression *call)
{
	// Evaluate arguments

	for (auto& argument : call->GetArguments())
	{
		argument->Accept(*this);
	}

	// Execute method and store result for the invocation

	auto result = Execute(call->GetMethod(), call->GetArguments());
	m_expressionMap.insert({call, result});
}

void Interpreter::Visit(HorseIR::Identifier *identifier)
{
	// Get the evaluated expression for the identifier

	m_expressionMap.insert({identifier, m_variableMap.at(identifier->GetString())});
}

void Interpreter::Visit(HorseIR::SymbolLiteral *literal)
{
	// Create a vector of symbols from the literal
	
	auto type = literal->GetType();
	if (!HorseIR::IsSymbolType(type))
	{
		Utils::Logger::LogError("Invalid type '" + type->ToString() + "' for symbol literal");
	}
	auto basicType = HorseIR::GetType<HorseIR::BasicType>(type);
	m_expressionMap.insert({literal, new Runtime::TypedDataVector<std::string>(basicType, literal->GetValues())});
}

}
