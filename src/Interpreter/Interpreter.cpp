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

#include "PTX/Program.h"

#include "Runtime/JITCompiler.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Interpreter {

void Interpreter::Execute(HorseIR::Program *program)
{
	Utils::Logger::LogSection("Starting program execution");

	m_program = program;

	HorseIR::EntryAnalysis entryAnalysis;
	entryAnalysis.Analyze(program);
	Execute(entryAnalysis.GetEntry(), {});
}

Runtime::DataObject *Interpreter::Execute(const HorseIR::Method *method, const std::vector<HorseIR::Expression *>& arguments)
{
	Utils::Logger::LogInfo("Executing method '" + method->GetName() + "'");

	if (method->IsKernel())
	{
		// Compile the HorseIR to PTX code using the current device

		auto& gpu = m_runtime.GetGPUManager();

		//TODO: Determine which part of the program should be sent to our compiler
		//TODO: This compiles the Builtins module too
		Runtime::JITCompiler compiler(gpu.GetCurrentDevice()->GetComputeCapability());
		PTX::Program *ptxProgram = compiler.Compile(m_program);

		// Optimize the generated PTX program

		if (Utils::Options::Get<>(Utils::Options::Opt_Optimize))
		{
			compiler.Optimize(ptxProgram);
		}

		// Create and load the CUDA module for the program

		CUDA::Module cModule = gpu.AssembleProgram(ptxProgram);

                // Fetch the handle to the GPU entry function

		CUDA::Kernel kernel(method->GetName(), arguments.size() + 2, cModule);

		auto timeExec_start = Utils::Chrono::Start();

		// Initialize buffers and kernel invocation

		//TODO: Logging of all steps and chrono
		//TODO: Get all data sizes from the table
		unsigned long geometrySize = 2048;

		CUDA::KernelInvocation invocation(kernel);

		//TODO: 512 should be chosen based on the device
		invocation.SetBlockShape(512, 1, 1);
		invocation.SetGridShape(geometrySize / 512, 1, 1);

		unsigned int index = 0;
		for (const auto& argument : arguments)
		{
			auto column = static_cast<Runtime::Vector *>(m_expressionMap.at(argument));

			Utils::Logger::LogInfo("test " + std::to_string(static_cast<int *>(column->GetData())[10]));

			//TODO: All buffers should be padded to a multiple of the thread count
			//TODO: Build a GPU buffer manager
			CUDA::Buffer buffer(column->GetData(), geometrySize * column->GetDataSize());
			buffer.AllocateOnGPU();
			buffer.TransferToGPU();

			invocation.SetParam(index++, buffer);
		}

		CUDA::TypedConstant<unsigned long> size(geometrySize);
		invocation.SetParam(index++, size);

		//TODO: Use dynamic information from the function to configure this
		//TODO: Use shape information to allocate the correct size
		auto type = new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Float32);
		std::vector<float> ret;
		ret.resize(1);
		auto retVec = new Runtime::TypedVector<float>(type, ret);
		CUDA::Buffer returnBuffer(retVec->GetData(), retVec->GetDataSize() * 1);

		invocation.SetParam(index++, returnBuffer);

		//TODO: Allocate the correct amount of dynamic shared memory based on the kernel
		// invocation.SetSharedMemorySize(sizeof(float) * 512);
		invocation.Launch();

		returnBuffer.TransferToCPU();

		auto timeExec = Utils::Chrono::End(timeExec_start);

		Utils::Logger::LogInfo("Kernel result = " + std::to_string(static_cast<float *>(retVec->GetData())[0]));

		return retVec;
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
		return nullptr;
	}
}

Runtime::DataObject *Interpreter::Execute(const HorseIR::BuiltinMethod *method, const std::vector<HorseIR::Expression *>& arguments)
{
	Utils::Logger::LogInfo("Executing builtin method '" + method->GetName() + "'");

	//TODO: Implement 2 last builtin methods
	//TODO: Update the table class to implement the spec of vector(names), list(columns)
	switch (method->GetKind())
	{
		case HorseIR::BuiltinMethod::Kind::Enlist:
			return nullptr;
		case HorseIR::BuiltinMethod::Kind::Table:
			return nullptr;
		case HorseIR::BuiltinMethod::Kind::ColumnValue:
		{
			auto table = static_cast<Runtime::Table *>(m_expressionMap.at(arguments.at(0)));
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

void Interpreter::Visit(HorseIR::CastExpression *cast)
{
	auto expression = cast->GetExpression();
	expression->Accept(*this);
	//TODO: Check cast is valid
	m_expressionMap.insert({cast, m_expressionMap.at(expression)});
}

void Interpreter::Visit(HorseIR::CallExpression *call)
{
	for (auto& argument : call->GetArguments())
	{
		argument->Accept(*this);
	}

	auto method = call->GetMethod();
	Runtime::DataObject *result = nullptr;
	switch (method->GetKind())
	{
		case HorseIR::MethodDeclaration::Kind::Definition:
			result = Execute(static_cast<HorseIR::Method *>(method), call->GetArguments());
			break;
		case HorseIR::MethodDeclaration::Kind::Builtin:
			result = Execute(static_cast<HorseIR::BuiltinMethod *>(method), call->GetArguments());
			break;
	}
	m_expressionMap.insert({call, result});
}

void Interpreter::Visit(HorseIR::Identifier *identifier)
{
	m_expressionMap.insert({identifier, m_variableMap.at(identifier->GetString())});
}

}
