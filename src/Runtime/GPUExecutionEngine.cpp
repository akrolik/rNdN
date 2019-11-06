#include "Runtime/GPUExecutionEngine.h"

#include <cmath>

#include "CUDA/Buffer.h"
#include "CUDA/Kernel.h"
#include "CUDA/KernelInvocation.h"
#include "CUDA/Module.h"

#include "PTX/Program.h"

#include "Analysis/DataObject/DataObjectAnalysis.h"
#include "Analysis/Geometry/GeometryAnalysis.h"
#include "Analysis/Geometry/KernelAnalysis.h"
#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeAnalysis.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Runtime/JITCompiler.h"
#include "Runtime/RuntimeUtils.h"
#include "Runtime/DataBuffers/BufferUtils.h"

#include "Utils/Logger.h"
#include "Utils/Math.h"
#include "Utils/Options.h"

namespace Runtime {

std::vector<DataBuffer *> GPUExecutionEngine::Execute(const HorseIR::Function *function, const Analysis::ShapeAnalysis& shapeAnalysis, const std::vector<DataBuffer *>& arguments)
{
	// Compile the HorseIR to PTX code using the current device

	auto& gpu = m_runtime.GetGPUManager();
	auto& device = gpu.GetCurrentDevice();

	Codegen::TargetOptions targetOptions;
	targetOptions.ComputeCapability = device->GetComputeCapability();
	targetOptions.MaxBlockSize = device->GetMaxThreadsDimension(0);
	targetOptions.WarpSize = device->GetWarpSize();

	const auto inputOptions = GetInputOptions(function, shapeAnalysis);

	// Initialize the geometry information for code generation

	JITCompiler compiler(targetOptions);
	const auto ptxProgram = compiler.Compile({function}, {&inputOptions});

	// Create and load the CUDA module for the program

	auto cModule = gpu.AssembleProgram(ptxProgram);

	// Compute the number of input arguments: function arguments + return arguments + [dynamic size arguments]

	unsigned int kernelArgumentCount = function->GetParameterCount() + function->GetReturnCount();

	// Dynamically sized kernel input parameter (we use this to select the correct load at runtime)

	for (const auto& parameter : inputOptions.ParameterShapes)
	{
		if (RuntimeUtils::IsDynamicDataShape(parameter.second, inputOptions.ThreadGeometry, false))
		{
			kernelArgumentCount++;
		}
	}

	// Dynamically sized kernel output parameters (we use this to indicate size of output buffers)

	for (const auto& shape : inputOptions.ReturnShapes)
	{
		// Any return argument that is not determined by the geometry or statically specified
		// needs a kernel argument for outputting the real size

		if (RuntimeUtils::IsDynamicDataShape(shape, inputOptions.ThreadGeometry, true))
		{
			kernelArgumentCount++;
		}
	}

	// Dynamically specify the thread geometry (vector: length; list: cells + cell lengths)

	if (const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(inputOptions.ThreadGeometry))
	{
		// [vector size]

		if (Analysis::ShapeUtils::IsDynamicSize(vectorGeometry->GetSize()))
		{
			kernelArgumentCount++;
		}
	}
	else if (Analysis::ShapeUtils::IsShape<Analysis::ListShape>(inputOptions.ThreadGeometry))
	{
		// [Cell threads] + list size + cell sizes

		if (inputOptions.ListCellThreads == Codegen::InputOptions::DynamicSize)
		{
			kernelArgumentCount++;
		}
		kernelArgumentCount += 2;
	}

	// Fetch the handle to the GPU entry function

	CUDA::Kernel kernel(function->GetName(), kernelArgumentCount, cModule);
	const auto& kernelOptions = ptxProgram->GetEntryFunction(function->GetName())->GetOptions();

	Utils::Logger::LogInfo("Generated program for function '" + function->GetName() + "' with options");
	Utils::Logger::LogInfo(kernelOptions.ToString(), 1);

	// Execute the compiled kernel on the GPU
	//   1. Create the invocation (thread sizes + arguments)
	//   2. Initialize the arguments
	//   3. Execute
	//   4. Initialize return values

	Utils::Logger::LogSection("Continuing program execution");

	// Initialize kernel invocation and configure the runtime thread layout

	CUDA::KernelInvocation invocation(kernel);

	// Collect runtime shape information for determining exact thread geometry and return shapes

	Analysis::DataObjectAnalysis runtimeDataAnalysis(m_program);
	Analysis::ShapeAnalysis runtimeShapeAnalysis(runtimeDataAnalysis, m_program, true);

	Analysis::DataObjectAnalysis::Properties inputObjects;
	Analysis::ShapeAnalysis::Properties inputShapes;
	for (auto i = 0u; i < arguments.size(); ++i)
	{
		const auto parameter = function->GetParameter(i);
		const auto symbol = parameter->GetSymbol();

		const auto object = shapeAnalysis.GetDataAnalysis().GetParameterObject(parameter);
		inputObjects[symbol] = object;

		// Setup compression constraints

		const auto symbolShape = shapeAnalysis.GetParameterShape(parameter);
		const auto runtimeShape = arguments.at(i)->GetShape();

		if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(symbolShape))
		{
			if (const auto vectorSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::CompressedSize>(vectorShape->GetSize()))
			{
				const auto runtimeVector = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(runtimeShape);
				runtimeShapeAnalysis.AddCompressionConstraint(vectorSize->GetPredicate(), runtimeVector->GetSize());
			}
		}
		inputShapes[symbol] = runtimeShape;
	}

	// Determine the thread geometry for the kernel

	runtimeDataAnalysis.Analyze(function, inputObjects);
	runtimeShapeAnalysis.Analyze(function, inputShapes);

	auto runtimeOptions = GetInputOptions(function, runtimeShapeAnalysis);

	Utils::Logger::LogInfo("Runtime Options");
	Utils::Logger::LogInfo(runtimeOptions.ToString(), 1);

	const auto [blockSize, blockCount] = GetBlockShape(runtimeOptions, targetOptions, kernelOptions);
	invocation.SetBlockShape(blockSize, 1, 1);
	invocation.SetGridShape(blockCount, 1, 1);

	// Initialize the input buffers for the kernel

	for (auto i = 0u; i < function->GetParameterCount(); ++i)
	{
		const auto parameter = function->GetParameter(i);
		const auto argument = arguments.at(i);

		Utils::Logger::LogInfo("Initializing input argument: " + parameter->GetName() + " [" + argument->Description() + "]");

		// Transfer the buffer to the GPU, for input parameters we assume read only

		auto buffer = argument->GetGPUReadBuffer();
		invocation.AddParameter(*buffer);

		// Allocate a size parameter if neded

		const auto inputShape = inputOptions.ParameterShapes.at(parameter);
		if (RuntimeUtils::IsDynamicDataShape(inputShape, inputOptions.ThreadGeometry, false))
		{
			const auto runtimeShape = runtimeOptions.ParameterShapes.at(parameter);
			AllocateSizeBuffer(invocation, runtimeShape, false);
		}
	}

	std::vector<DataBuffer *> returnBuffers;
	std::vector<CUDA::Buffer *> sizeBuffers;
	for (auto i = 0u; i < function->GetReturnCount(); ++i)
	{
		// Create a new buffer for the return value

		const auto type = function->GetReturnType(i);
		const auto shape = runtimeOptions.ReturnShapes.at(i);
		auto buffer = DataBuffer::Create(type, shape);

		Utils::Logger::LogInfo("Initializing return argument: " + std::to_string(i) + " [" + buffer->Description() + "]");

		// Transfer the write buffer to te GPU, we assume all returns write (or else...)

		returnBuffers.push_back(buffer);
		invocation.AddParameter(*buffer->GetGPUWriteBuffer());

		// Allocate a size parameter if neded

		const auto inputShape = inputOptions.ReturnShapes.at(i);
		if (RuntimeUtils::IsDynamicDataShape(inputShape, inputOptions.ThreadGeometry, true))
		{
			const auto runtimeShape = runtimeOptions.ReturnShapes.at(i);
			sizeBuffers.push_back(AllocateSizeBuffer(invocation, runtimeShape, true));
		}
	}

	// Setup constant dynamic size parameters and allocate the dynamic shared memory according to the kernel

	if (const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(runtimeOptions.ThreadGeometry))
	{
		const auto inputGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(inputOptions.ThreadGeometry);
		if (Analysis::ShapeUtils::IsDynamicSize(inputGeometry->GetSize()))
		{
			if (const auto constantSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(vectorGeometry->GetSize()))
			{
				AllocateConstantParameter(invocation, constantSize->GetValue(), "<vector geometry>");
			}
			else
			{
				Utils::Logger::LogError("Invocation thread geometry must be static " + Analysis::ShapeUtils::ShapeString(vectorGeometry));
			}
		}

		invocation.SetDynamicSharedMemorySize(kernelOptions.GetDynamicSharedMemorySize());
	}
	else if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(runtimeOptions.ThreadGeometry))
	{
		// Add the dynamic thread count to the parameters if needed

		if (inputOptions.ListCellThreads == Codegen::InputOptions::DynamicSize)
		{
			AllocateConstantParameter(invocation, runtimeOptions.ListCellThreads, "<geometry cell threads>");
		}

		// Allocate a buffer with the cell sizes for the execution geometry

		const auto inputListShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(inputOptions.ThreadGeometry);
		if (Analysis::ShapeUtils::IsDynamicSize(inputListShape->GetListSize()))
		{
			if (const auto constantSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(listShape->GetListSize()))
			{
				AllocateConstantParameter(invocation, constantSize->GetValue(), "<geometry list size>");
			}
			else
			{
				Utils::Logger::LogError("Invocation thread geometry must be static " + Analysis::ShapeUtils::ShapeString(vectorGeometry));
			}
		}

		// Always transfer the cell sizes

		AllocateCellSizes(invocation, listShape, "<geometry cell sizes>");

		//TODO: Determine the correct amount of shared memory for cells
		invocation.SetDynamicSharedMemorySize(kernelOptions.GetDynamicSharedMemorySize() * 2);
	}

	// Launch the kernel!

	invocation.Launch();

	std::vector<DataBuffer *> resizedBuffers;
	auto resizeIndex = 0u;
	auto returnIndex = 0u;
	for (auto returnBuffer : returnBuffers)
	{
		// Allocate a size parameter if neded

		const auto inputShape = inputOptions.ReturnShapes.at(returnIndex++);
		if (RuntimeUtils::IsDynamicDataShape(inputShape, inputOptions.ThreadGeometry, true))
		{
			if (const auto vectorBuffer = BufferUtils::GetBuffer<VectorBuffer>(returnBuffer))
			{
				auto size = 0u;

				// Transfer the size to the CPU to resize the buffer

				auto sizeBuffer = sizeBuffers.at(resizeIndex++);
				sizeBuffer->SetCPUBuffer(&size);
				sizeBuffer->TransferToCPU();

				//TODO: Temporary hack
				if (size == 0)
				{
					resizedBuffers.push_back(returnBuffer);
					continue;
				}

				// Allocate a new buffer and transfer the contents

				auto resizedBuffer = VectorBuffer::Create(vectorBuffer->GetType(), new Analysis::Shape::ConstantSize(size));
				CUDA::Buffer::Copy(resizedBuffer->GetGPUWriteBuffer(), vectorBuffer->GetGPUReadBuffer(), size * vectorBuffer->GetElementSize());

				Utils::Logger::LogInfo("Resized dynamic buffer " + returnBuffer->Description() + " to " + resizedBuffer->Description());

				resizedBuffers.push_back(resizedBuffer);
				delete returnBuffer;
			}
			else
			{
				Utils::Logger::LogError("Unable to resize dynamically sized return buffer " + returnBuffer->Description());
			}
		}
		else
		{
			resizedBuffers.push_back(returnBuffer);
		}
		resizeIndex++;
	}

	return resizedBuffers;
}

Codegen::InputOptions GPUExecutionEngine::GetInputOptions(const HorseIR::Function *function, const Analysis::ShapeAnalysis& shapeAnalysis) const
{
	Analysis::GeometryAnalysis geometryAnalysis(shapeAnalysis);
	geometryAnalysis.Analyze(function);

	Analysis::KernelAnalysis kernelAnalysis(geometryAnalysis);
	kernelAnalysis.Analyze(function);

	// Initialize the geometry information for code generation

	Codegen::InputOptions inputOptions;
	inputOptions.ThreadGeometry = kernelAnalysis.GetOperatingGeometry();
	//TODO: Determine order
	inputOptions.InOrderBlocks = Analysis::ShapeUtils::IsShape<Analysis::VectorShape>(inputOptions.ThreadGeometry);

	const auto& dataAnalysis = shapeAnalysis.GetDataAnalysis();

	for (const auto& parameter : function->GetParameters())
	{
		inputOptions.Parameters[parameter->GetSymbol()] = parameter;

		inputOptions.ParameterShapes[parameter] = shapeAnalysis.GetParameterShape(parameter);
		inputOptions.ParameterObjects[dataAnalysis.GetParameterObject(parameter)] = parameter;
	}
	inputOptions.ReturnShapes = shapeAnalysis.GetReturnShapes();

	// Specify the number of threads for each cell computation in list thread geometry

	if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(inputOptions.ThreadGeometry))
	{
		if (Analysis::ShapeUtils::IsDynamicShape(listShape))
		{
			inputOptions.ListCellThreads = Codegen::InputOptions::DynamicSize;
		}
		else
		{
			const auto cellSizes = GetCellSizes(listShape);
			const auto averageCellSize = std::accumulate(cellSizes.begin(), cellSizes.end(), 0) / cellSizes.size();
			inputOptions.ListCellThreads = Utils::Math::Power2(averageCellSize);
		}
	}

	return inputOptions;
}

std::vector<std::uint32_t> GPUExecutionEngine::GetCellSizes(const Analysis::ListShape *shape) const
{
	// Form a vector of cell sizes, for lists of constan-sized vectors

	std::vector<std::uint32_t> cellSizes;
	for (const auto cellShape : shape->GetElementShapes())
	{
		if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellShape))
		{
			if (const auto constantSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(vectorShape->GetSize()))
			{
				cellSizes.push_back(constantSize->GetValue());
				continue;
			}
		}
		Utils::Logger::LogError("Unable to get constant cell sizes for list shape " + Analysis::ShapeUtils::ShapeString(shape));
	}
	return cellSizes;
}

std::pair<unsigned int, unsigned int> GPUExecutionEngine::GetBlockShape(Codegen::InputOptions& runtimeOptions, const Codegen::TargetOptions& targetOptions, const PTX::FunctionOptions& kernelOptions) const
{
	// Compute the block size and count based on the kernel, input and target configurations
	// We assume that all sizes are known at this point

	const auto threadGeometry = runtimeOptions.ThreadGeometry;
	if (const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(threadGeometry))
	{
		if (const auto constantSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(vectorGeometry->GetSize()))
		{
			auto size = constantSize->GetValue();
			auto blockSize = kernelOptions.GetBlockSize();

			if (blockSize == PTX::FunctionOptions::DynamicBlockSize)
			{
				// Fill the multiprocessors, but not more than the data size

				if (size < targetOptions.MaxBlockSize)
				{
					blockSize = size;
				}
				else
				{
					blockSize = targetOptions.MaxBlockSize;
				}

				if (kernelOptions.GetThreadMultiple() != 0)
				{
					// Maximize the block size based on the GPU and thread multiple

					auto multiple = kernelOptions.GetThreadMultiple();
					blockSize = ((blockSize + multiple - 1) / multiple) * multiple;
				}
			}

			auto blockCount = ((size + blockSize - 1) / blockSize);
			return {blockSize, blockCount};
		}
	}
	else if (const auto listGeometry = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(threadGeometry))
	{
		if (const auto constantSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(listGeometry->GetListSize()))
		{
			auto cellCount = constantSize->GetValue();
			auto cellSize = kernelOptions.GetBlockSize();

			// Check if the cell size is specified as constant or dynamic

			if (cellSize == PTX::FunctionOptions::DynamicBlockSize)
			{
				// The thread number was not specified in the input or kernel properties, but determined
				// at runtime depending on the cell sizes

				cellSize = runtimeOptions.ListCellThreads;

				if (kernelOptions.GetThreadMultiple() != 0)
				{
					// Ensure the thread number is a multiple of the kernel specification

					auto multiple = kernelOptions.GetThreadMultiple();
					cellSize = ((cellSize + multiple - 1) / multiple) * multiple;
				}
			}

			// Important: Update the number of cells per thread

			runtimeOptions.ListCellThreads = cellSize;

			// Compute the block size, capped by the total thread count if smaller

			auto blockSize = targetOptions.MaxBlockSize;
			if (blockSize > cellSize * cellCount)
			{
				blockSize = cellSize * cellCount;
			}
			auto blockCount = ((cellSize * cellCount + blockSize - 1) / blockSize);
			return {blockSize, blockCount};
		}
	}

	Utils::Logger::LogError("Unknown block shape for thread geometry " + Analysis::ShapeUtils::ShapeString(threadGeometry));
}

void GPUExecutionEngine::AllocateConstantParameter(CUDA::KernelInvocation& invocation, std::uint32_t value, const std::string& description) const
{
	Utils::Logger::LogInfo("Initializing constant input argument: " + description + " [i32(" + std::to_string(sizeof(std::uint32_t)) + " bytes) = " + std::to_string(value) + "]");

	auto sizeConstant = new CUDA::TypedConstant<std::uint32_t>(value);
	invocation.AddParameter(*sizeConstant);
}

CUDA::Buffer *GPUExecutionEngine::AllocateCellSizes(CUDA::KernelInvocation& invocation, const Analysis::ListShape *shape, const std::string& description) const
{
	// Collect the cell sizes array

	auto cellSizes = GetCellSizes(shape);

	Utils::Logger::LogInfo("Initializing input argument: " + description + " [i32(" + std::to_string(sizeof(std::uint32_t)) + " bytes) x " + std::to_string(cellSizes.size()) + "]");

	// Transfer to the GPU

	auto buffer = new CUDA::Buffer(cellSizes.data(), cellSizes.size() * sizeof(std::uint32_t));
	buffer->AllocateOnGPU();
	buffer->TransferToGPU();
	invocation.AddParameter(*buffer);
	return buffer;
}

CUDA::Buffer *GPUExecutionEngine::AllocateSizeBuffer(CUDA::KernelInvocation& invocation, const Analysis::Shape *shape, bool returnParameter) const
{
	// Allocate a size buffer for the shape

	if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(shape))
	{
		if (!returnParameter)
		{
			if (const auto constantSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(vectorShape->GetSize()))
			{
				// Constant size buffers can be directly added to the kernel parameters

				AllocateConstantParameter(invocation, constantSize->GetValue(), "<vector size>");
				return nullptr;
			}
		}

		// This requires a global memory allocation to output the result

		Utils::Logger::LogInfo("Initializing input argument: <dynamic vector size> [i32(" + std::to_string(sizeof(std::uint32_t)) + " bytes) x 1]");

		auto buffer = new CUDA::Buffer(sizeof(std::uint32_t));
		buffer->AllocateOnGPU();
		buffer->Clear();
		invocation.AddParameter(*buffer);

		return buffer;
	}
	else if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
	{
		// A list size is given by the cell sizes (the number of cells is specified through another parameter)

		return AllocateCellSizes(invocation, listShape, "<dynamic cell sizes>");
	}
	else
	{
		Utils::Logger::LogError("Unable to allocate size buffer for " + Analysis::ShapeUtils::ShapeString(shape) + "[unsupported shape]");
	}
}

}
