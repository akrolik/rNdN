#include "Interpreter/Interpreter.h"

#include "CUDA/Buffer.h"
#include "CUDA/Kernel.h"
#include "CUDA/KernelInvocation.h"
#include "CUDA/Module.h"

#include "HorseIR/EntryAnalysis.h"
#include "HorseIR/Tree/BuiltinMethod.h"
#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"
#include "HorseIR/Tree/Expressions/CastExpression.h"
#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/Symbol.h"
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

Runtime::DataObject *Interpreter::Execute(const HorseIR::MethodDeclaration *method, const std::vector<HorseIR::Expression *>& arguments)
{
	switch (method->GetKind())
	{
		case HorseIR::MethodDeclaration::Kind::Definition:
			return Execute(static_cast<const HorseIR::Method *>(method), arguments);
		case HorseIR::MethodDeclaration::Kind::Builtin:
			return Execute(static_cast<const HorseIR::BuiltinMethod *>(method), arguments);
	}

	Utils::Logger::LogError("Cannot execute method '" + method->GetName() + "'");
}

Runtime::DataObject *Interpreter::Execute(const HorseIR::Method *method, const std::vector<HorseIR::Expression *>& arguments)
{
	Utils::Logger::LogInfo("Executing method '" + method->GetName() + "'");

	if (method->IsKernel())
	{
		// Compile the HorseIR to PTX code using the current device

		auto& gpu = m_runtime.GetGPUManager();
		auto& device =  gpu.GetCurrentDevice();

		Codegen::TargetOptions targetOptions;
		targetOptions.ComputeCapability = device->GetComputeCapability();
		targetOptions.MaxBlockSize = device->GetMaxThreadsDimension(0);
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

                // Fetch the handle to the GPU entry function

		CUDA::Kernel kernel(method->GetName(), arguments.size() + 2, cModule);

		Utils::Logger::LogSection("Continuing program execution");

		auto timeExec_start = Utils::Chrono::Start();

		// Initialize buffers and kernel invocation

		CUDA::KernelInvocation invocation(kernel);

		invocation.SetBlockShape(targetOptions.MaxBlockSize, 1, 1);
		invocation.SetGridShape((inputOptions.InputSize - 1) / targetOptions.MaxBlockSize + 1, 1, 1);

		unsigned int index = 0;
		for (const auto& argument : arguments)
		{
			auto column = static_cast<Runtime::DataVector *>(m_expressionMap.at(argument));

			Utils::Logger::LogInfo("Transferring input argument '" + argument->ToString() + "' [" + column->GetType()->ToString() + "(" + std::to_string(column->GetElementSize()) + " bytes) x " + std::to_string(column->GetElementCount()) + "]");

			//TODO: All buffers should be padded to a multiple of the thread count
			//TODO: Build a GPU buffer manager
			auto buffer = new CUDA::Buffer(column->GetData(), column->GetDataSize());

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
			case HorseIR::Type::Kind::Primitive:
				//TODO: Use shape information to allocate the correct size
				returnData = Runtime::DataVector::CreateVector(static_cast<const HorseIR::PrimitiveType *>(returnType), 1);
				break;
			defualt:
				Utils::Logger::LogError("Unsupported return type " + returnType->ToString());
		}

		Utils::Logger::LogInfo("Initializing return argument [" + returnType->ToString() + "(" + std::to_string(returnData->GetElementSize()) + " bytes) x " + std::to_string(returnData->GetElementCount()) + "]");

		CUDA::Buffer returnBuffer(returnData->GetData(), returnData->GetDataSize());
		returnBuffer.AllocateOnGPU();
		returnBuffer.TransferToGPU();
		invocation.SetParameter(index++, returnBuffer);

		//TODO: Allocate the correct amount of dynamic shared memory based on the kernel
		// invocation.SetSharedMemorySize(sizeof(float) * 512);
		invocation.Launch();

		returnBuffer.TransferToCPU();

		auto timeExec = Utils::Chrono::End(timeExec_start);

		Utils::Logger::LogTiming("Kernel execution", timeExec);
		Utils::Logger::LogInfo("Kernel result = " + std::to_string(static_cast<float *>(returnData->GetData())[0]));

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

Runtime::DataObject *Interpreter::Execute(const HorseIR::BuiltinMethod *method, const std::vector<HorseIR::Expression *>& arguments)
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
			auto columnName = static_cast<const HorseIR::Symbol *>(arguments.at(1))->GetName();

			return table->GetColumn(columnName);
		}
		case HorseIR::BuiltinMethod::Kind::LoadTable:
		{
			auto dataRegistry = m_runtime.GetDataRegistry();
			auto tableName = static_cast<const HorseIR::Symbol *>(arguments.at(0))->GetName();

			return dataRegistry.GetTable(tableName);
		}
		default:
			Utils::Logger::LogError("Builtin function '" + method->GetName() + "' not implemented");
	}
}

void Interpreter::Visit(HorseIR::AssignStatement *assign)
{
	auto expression = assign->GetExpression();
	expression->Accept(*this);
	m_variableMap.insert({assign->GetTargetName(), m_expressionMap.at(expression)});
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

	// Check the cast is valid

	auto result = m_expressionMap.at(expression);

	auto expressionType = expression->GetType();
	auto resultType = result->GetType();

	//TODO: Remove this hack
	if (expressionType == nullptr)
	{
		expressionType = new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Int32);
	}

	if (*expressionType != *resultType)
	{
		Utils::Logger::LogError("Invalid cast, " + expressionType->ToString() + " cannot be cast to " + resultType->ToString());
	}

	m_expressionMap.insert({cast, result});
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

void Interpreter::Visit(HorseIR::Symbol *symbol)
{
	// Create a vector of symbols from the literal

	m_expressionMap.insert({symbol, new Runtime::TypedDataVector<std::string>(static_cast<const HorseIR::PrimitiveType *>(symbol->GetType()), {symbol->GetName()})});
}

}
