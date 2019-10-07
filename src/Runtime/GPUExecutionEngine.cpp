#include "Runtime/GPUExecutionEngine.h"

#include "CUDA/Buffer.h"
#include "CUDA/Kernel.h"
#include "CUDA/KernelInvocation.h"
#include "CUDA/Module.h"

#include "PTX/Program.h"

#include "Analysis/Geometry/GeometryAnalysis.h"
#include "Analysis/Geometry/KernelAnalysis.h"
#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeAnalysis.h"
#include "Analysis/Shape/ShapeUtils.h"

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

	// Collect shape information for input arguments

	//GLOBAL: Collect shape information
	Analysis::ShapeAnalysis::Properties inputShapes;
	for (auto i = 0u; i < arguments.size(); ++i)
	{
		const auto symbol = function->GetParameter(i)->GetSymbol();
		const auto shape = arguments.at(i)->GetShape();
		inputShapes[symbol] = shape;
	}

	// Determine the thread geometry for the kernel

	Analysis::ShapeAnalysis shapeAnalysis(m_program);
	shapeAnalysis.Analyze(function, inputShapes);

	Analysis::GeometryAnalysis geometryAnalysis(shapeAnalysis);
	geometryAnalysis.Analyze(function);

	Analysis::KernelAnalysis kernelAnalysis(geometryAnalysis);
	kernelAnalysis.Analyze(function);

	auto threadGeometry = kernelAnalysis.GetThreadGeometry();
	Utils::Logger::LogInfo("Thread geometry: " + threadGeometry->ToString());

	Codegen::InputOptions inputOptions;
	switch (threadGeometry->GetKind())
	{
		case Analysis::ThreadGeometry::Kind::Vector:
		{
			inputOptions.ActiveThreads = threadGeometry->GetSize();
			break;
		}
		case Analysis::ThreadGeometry::Kind::List:
		{
			inputOptions.ActiveBlocks = threadGeometry->GetSize();
			break;
		}
	}

	JITCompiler compiler(targetOptions, inputOptions);
	auto ptxProgram = compiler.Compile({function});

	// Optimize the generated PTX program

	if (Utils::Options::Get<>(Utils::Options::Opt_Optimize))
	{
		compiler.Optimize(ptxProgram);
	}

	// Create and load the CUDA module for the program

	auto cModule = gpu.AssembleProgram(ptxProgram);

	// Compute the number of input arguments: function arguments + return arguments

	unsigned int kernelArgumentCount = function->GetParameterCount() + function->GetReturnCount();

	// Fetch the handle to the GPU entry function

	CUDA::Kernel kernel(function->GetName(), kernelArgumentCount, cModule);
	auto& kernelOptions = ptxProgram->GetEntryFunction(function->GetName())->GetOptions();

	Utils::Logger::LogInfo("Generated program for function '" + function->GetName() + "' with options");
	Utils::Logger::LogInfo(kernelOptions.ToString(), 1);

	// Execute the compiled kernel on the GPU
	//   1. Create the invocation (thread sizes + arguments)
	//   2. Initialize the arguments
	//   3. Execute
	//   4. Initialize return values

	Utils::Logger::LogSection("Continuing program execution");

	// Initialize kernel invocation with the thread options

	auto [blockSize, blockCount] = GetBlockShape(inputOptions, targetOptions, kernelOptions);

	CUDA::KernelInvocation invocation(kernel);
	invocation.SetBlockShape(blockSize, 1, 1);
	invocation.SetGridShape(blockCount, 1, 1);

	// Initialize the input buffers for the kernel

	auto paramIndex = 0u;
	for (const auto& argument : arguments)
	{
		Utils::Logger::LogInfo("Initializing input argument [" + argument->Description() + "]");

		// Transfer the buffer to the GPU, for input parameters we assume read only

		auto buffer = argument->GetGPUReadBuffer();
		invocation.SetParameter(paramIndex++, *buffer);
	}

	std::vector<DataBuffer *> returnBuffers;
	for (auto i = 0u; i < function->GetReturnCount(); ++i)
	{
		// Create a new buffer for the return value

		const auto returnType = function->GetReturnType(i);
		const auto returnShape = shapeAnalysis.GetReturnShape(i);
		auto returnBuffer = DataBuffer::Create(returnType, returnShape);

		Utils::Logger::LogInfo("Initializing return argument [" + returnBuffer->Description() + "]");

		returnBuffers.push_back(returnBuffer);
		invocation.SetParameter(paramIndex++, *returnBuffer->GetGPUWriteBuffer());
	}

	// Configure the dynamic shared memory according to the kernel

	invocation.SetSharedMemorySize(kernelOptions.GetSharedMemorySize());

	// Launch the kernel!

	invocation.Launch();

	return returnBuffers;
}

std::pair<unsigned int, unsigned int> GPUExecutionEngine::GetBlockShape(const Codegen::InputOptions& inputOptions, const Codegen::TargetOptions& targetOptions, const PTX::FunctionOptions& kernelOptions) const
{
	// Compute the block size and count based on the kernel, input and target configurations

	auto blockSize = kernelOptions.GetBlockSize();
	if (blockSize == PTX::FunctionOptions::DynamicBlockSize)
	{
		if (kernelOptions.GetThreadMultiple() != 0)
		{
			// Maximize the block size based on the GPU and thread multiple

			auto multiple = kernelOptions.GetThreadMultiple();
			if (inputOptions.ActiveThreads != Codegen::InputOptions::DynamicSize && inputOptions.ActiveThreads < targetOptions.MaxBlockSize)
			{
				blockSize = (multiple * (inputOptions.ActiveThreads / multiple));
			}
			blockSize = (multiple * (targetOptions.MaxBlockSize / multiple));
		}
		else
		{
			// Fill the multiprocessors, but not more than the data size

			if (inputOptions.ActiveThreads != Codegen::InputOptions::DynamicSize && inputOptions.ActiveThreads < targetOptions.MaxBlockSize)
			{
				blockSize = inputOptions.ActiveThreads;
			}
			else
			{
				blockSize = targetOptions.MaxBlockSize;
			}
		}
	}

	if (inputOptions.ActiveThreads == Codegen::InputOptions::DynamicSize)
	{
		auto blockCount = inputOptions.ActiveBlocks;
		return {blockSize, blockCount};
	}
	else
	{
		auto blockCount = ((inputOptions.ActiveThreads - 1) / blockSize + 1);
		return {blockSize, blockCount};
	}
}

}
