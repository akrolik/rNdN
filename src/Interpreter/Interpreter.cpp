#include "Interpreter/Interpreter.h"

#include "CUDA/Kernel.h"
#include "CUDA/Module.h"

#include "HorseIR/EntryAnalysis.h"
#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Expressions/CallExpression.h"
#include "HorseIR/Tree/Expressions/CastExpression.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"

#include "PTX/Program.h"

#include "Runtime/JITCompiler.h"

#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Interpreter {

void Interpreter::Execute(HorseIR::Program *program)
{
	Utils::Logger::LogSection("Starting program execution");

	m_program = program;

	HorseIR::EntryAnalysis entryAnalysis;
	entryAnalysis.Analyze(program);
	Execute(entryAnalysis.GetEntry());
}

void Interpreter::Execute(HorseIR::Method *method)
{
	Utils::Logger::LogInfo("Executing method '" + method->GetName() + "'");

	if (method->IsKernel())
	{
		// Compile the HorseIR to PTX code using the current device

		auto& gpu = m_runtime.GetGPUManager();

		Runtime::JITCompiler compiler(gpu.GetCurrentDevice()->GetComputeCapability());
		PTX::Program *ptxProgram = compiler.Compile(m_program);

		// Optimize the generated PTX program

		if (Utils::Options::Get<>(Utils::Options::Opt_Optimize))
		{
			compiler.Optimize(ptxProgram);
		}

		//TODO: Output optimized PTX code

		// Create and load the CUDA module for the program

		CUDA::Module cModule = gpu.AssembleProgram(ptxProgram);

                // Fetch the handle to the GPU entry function

		CUDA::Kernel kernel(method->GetName(), 0, cModule);

		//TODO: Implement scheduler and execution engine

		// auto timeExec_start = Utils::Chrono::Start();
		// Initialize buffers and kernel invocation

		// CUDA::Buffer bufferA(dataA, size);
		// CUDA::TypedConstant<unsigned long> valueB(numElements);
		// // CUDA::Buffer bufferB(&numElements, sizeof(unsigned long));
		// CUDA::Buffer bufferC(dataC, sizeof(float));
		// bufferA.AllocateOnGPU(); bufferA.TransferToGPU();
		// // bufferB.AllocateOnGPU(); bufferB.TransferToGPU();
		// bufferC.AllocateOnGPU();

		// CUDA::KernelInvocation invocation(kernel);
		// invocation.SetBlockShape(512, 1, 1);
		// invocation.SetGridShape(2, 1, 1);
		// invocation.SetParam(0, bufferA);
		// invocation.SetParam(1, valueB);
		// // invocation.SetParam(1, bufferB);
		// invocation.SetParam(2, bufferC);
		// invocation.SetSharedMemorySize(sizeof(float) * 512);
		// invocation.Launch();

		// bufferC.TransferToCPU();

		// auto timeExec = Utils::Chrono::End(timeExec_start);
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
	}
}

void Interpreter::Visit(HorseIR::AssignStatement *assign)
{
	assign->GetExpression()->Accept(*this);
	m_dataMap.insert({assign->GetTargetName(), m_expressionMap.at(assign->GetExpression())});
}

void Interpreter::Visit(HorseIR::CastExpression *cast)
{
	cast->GetExpression()->Accept(*this);
	m_expressionMap.insert({cast, nullptr});
}

void Interpreter::Visit(HorseIR::CallExpression *call)
{
	//TODO: Implement a global symbol table for the compilation unit
	// for (auto& module : m_program->GetModules())
	// {
	// 	for (auto& c : module->GetContents())
	// 	{
	// 		HorseIR::Method *m = nullptr;
	// 		if (m = dynamic_cast<HorseIR::Method *>(c))
	// 		{
	// 			if (m->GetName() == call->GetName())
	// 			{
	// 				Execute(m);
	// 			}
	// 		}
	// 	}
	// }
	m_expressionMap.insert({call, nullptr});
}

}
