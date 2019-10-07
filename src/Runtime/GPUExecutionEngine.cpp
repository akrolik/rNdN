#include "Runtime/GPUExecutionEngine.h"

#include "CUDA/Buffer.h"
#include "CUDA/Kernel.h"
#include "CUDA/KernelInvocation.h"
#include "CUDA/Module.h"

#include "PTX/Program.h"

#include "Runtime/JITCompiler.h"

#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Runtime {

std::vector<DataBuffer *> GPUExecutionEngine::Execute(const HorseIR::Function *function, const std::vector<DataBuffer *>& arguments)
{
	// Compile the HorseIR to PTX code using the current device

	auto& gpu = m_runtime.GetGPUManager();
	auto& device = gpu.GetCurrentDevice();

	Codegen::TargetOptions targetOptions;
	targetOptions.ComputeCapability = device->GetComputeCapability();
	targetOptions.MaxBlockSize = device->GetMaxThreadsDimension(0);
	targetOptions.WarpSize = device->GetWarpSize();

	//TODO: Get the input geometry size from the table
	Codegen::InputOptions inputOptions;
	inputOptions.InputSize = 6001215;

	JITCompiler compiler(targetOptions, inputOptions);
	auto ptxProgram = compiler.Compile({function});

	// Optimize the generated PTX program

	if (Utils::Options::Get<>(Utils::Options::Opt_Optimize))
	{
		compiler.Optimize(ptxProgram);
	}

	// Create and load the CUDA module for the program

	auto cModule = gpu.AssembleProgram(ptxProgram);

	// Compute the number of input arguments: function arguments + return arguments + (dynamic size)

	unsigned int kernelArgumentCount = function->GetParameterCount() + function->GetReturnCount();
	if (inputOptions.InputSize == Codegen::InputOptions::DynamicSize)
	{
		kernelArgumentCount++;
	}

	// Fetch the handle to the GPU entry function

	CUDA::Kernel kernel(function->GetName(), kernelArgumentCount, cModule);
	auto& kernelOptions = ptxProgram->GetEntryFunction(function->GetName())->GetOptions();

	Utils::Logger::LogInfo("Generated program for function '" + function->GetName() + "' with options");
	Utils::Logger::LogInfo(kernelOptions.ToString(), 1);

	// Execute the compiled kernel on the GPU
	//   1. Create the invocation (thread sizes + arguments)
	//   2. Transfer the arguments
	//   3. Execute
	//   4. Transfer return value

	Utils::Logger::LogSection("Continuing program execution");

	auto timeExec_start = Utils::Chrono::Start();

	// Initialize kernel invocation with the thread options

	auto blockSize = GetBlockSize(inputOptions, targetOptions, kernelOptions);

	CUDA::KernelInvocation invocation(kernel);
	invocation.SetBlockShape(blockSize, 1, 1);
	invocation.SetGridShape((inputOptions.InputSize - 1) / blockSize + 1, 1, 1);

	// Initialize the input buffers for the kernel

	auto paramIndex = 0u;
	for (const auto& argument : arguments)
	{
		Utils::Logger::LogInfo("Initializing input argument [" + argument->Description() + "]");

		// Transfer the buffer to the GPU, for input parameters we assume read only

		auto buffer = argument->GetGPUReadBuffer();
		invocation.SetParameter(paramIndex++, *buffer);
	}

	// Add the dynamic size parameter if needed

	if (inputOptions.InputSize == Codegen::InputOptions::DynamicSize)
	{
		//TODO: This just uses the dynamic size constant and not the actual size!
		CUDA::TypedConstant<uint64_t> sizeConstant(inputOptions.InputSize);
		invocation.SetParameter(paramIndex++, sizeConstant);
	}

	std::vector<DataBuffer *> returnBuffers;
	for (const auto& returnType : function->GetReturnTypes())
	{
		//TODO: Determine size
		auto returnBuffer = VectorBuffer::Create(static_cast<HorseIR::BasicType *>(returnType), 1);
		returnBuffers.push_back(returnBuffer);

		Utils::Logger::LogInfo("Initializing return argument [" + returnBuffer->Description() + "]");

		auto gpuBuffer = returnBuffer->GetGPUWriteBuffer();
		invocation.SetParameter(paramIndex++, *gpuBuffer);
	}

	// Configure the dynamic shared memory according to the kernel

	invocation.SetSharedMemorySize(kernelOptions.GetSharedMemorySize());

	// Launch the kernel!

	invocation.Launch();

	return returnBuffers;
}

unsigned int GPUExecutionEngine::GetBlockSize(const Codegen::InputOptions& inputOptions, const Codegen::TargetOptions& targetOptions, const PTX::FunctionOptions& kernelOptions) const
{
	// Compute the number of threads based on the kernel and target configurations

	auto kernelThreadCount = kernelOptions.GetThreadCount();
	if (kernelThreadCount == PTX::FunctionOptions::DynamicThreadCount)
	{
		if (kernelOptions.GetThreadMultiple() != 0)
		{
			// Maximize the thread count based on the GPU and thread multiple

			auto multiple = kernelOptions.GetThreadMultiple();
			if (inputOptions.InputSize < targetOptions.MaxBlockSize)
			{
				return (multiple * (inputOptions.InputSize / multiple));
			}
			return (multiple * (targetOptions.MaxBlockSize / multiple));
		}
		
		// Fill the multiprocessors, but not more than the data size

		if (inputOptions.InputSize < targetOptions.MaxBlockSize)
		{
			return inputOptions.InputSize;
		}

		return targetOptions.MaxBlockSize;
	}

	// Fixed number of threads

	if (kernelThreadCount > targetOptions.MaxBlockSize)
	{
		Utils::Logger::LogError("Thread count " + std::to_string(kernelThreadCount) + " is not supported by target (MaxBlockSize= " + std::to_string(targetOptions.MaxBlockSize));
	}
	return kernelThreadCount;
}

}
