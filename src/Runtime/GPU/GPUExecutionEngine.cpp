#include "Runtime/GPU/GPUExecutionEngine.h"

#include "CUDA/Buffer.h"
#include "CUDA/Kernel.h"
#include "CUDA/KernelInvocation.h"

#include "Analysis/DataObject/DataObjectAnalysis.h"
#include "Analysis/DataObject/DataCopyAnalysis.h"
#include "Analysis/Geometry/KernelOptionsAnalysis.h"
#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeAnalysis.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Runtime/RuntimeUtils.h"
#include "Runtime/DataBuffers/BufferUtils.h"

#include "Utils/Logger.h"

namespace Runtime {

std::vector<DataBuffer *> GPUExecutionEngine::Execute(const HorseIR::Function *function, const std::vector<DataBuffer *>& arguments)
{
	// Get the input options used for codegen
	
	const auto program = m_runtime.GetGPUManager().GetProgram();

	const auto kernelName = function->GetName();
	const auto& kernelOptions = program->GetKernelOptions(kernelName);

	const auto inputOptions = kernelOptions.GetCodegenOptions();

	// Collect runtime shape information for determining exact thread geometry and return shapes

	Analysis::DataObjectAnalysis dataAnalysis(m_program);
	Analysis::ShapeAnalysis shapeAnalysis(dataAnalysis, m_program, true);

	Analysis::DataObjectAnalysis::Properties inputObjects;
	Analysis::ShapeAnalysis::Properties inputShapes;
	for (auto i = 0u; i < arguments.size(); ++i)
	{
		const auto parameter = function->GetParameter(i);
		const auto symbol = parameter->GetSymbol();

		const auto object = inputOptions->ParameterObjects.at(parameter);
		const auto runtimeObject = new Analysis::DataObject(object->GetObjectID(), arguments.at(i));
		inputObjects[symbol] = runtimeObject;

		// Setup compression constraints

		const auto symbolShape = inputOptions->ParameterShapes.at(parameter);
		const auto runtimeShape = arguments.at(i)->GetShape();

		if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(symbolShape))
		{
			if (const auto vectorSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::CompressedSize>(vectorShape->GetSize()))
			{
				const auto runtimeVector = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(runtimeShape);
				shapeAnalysis.AddCompressionConstraint(vectorSize->GetPredicate(), runtimeVector->GetSize());
			}
		}
		inputShapes.first[symbol] = runtimeShape;
		inputShapes.second[symbol] = runtimeShape;
	}

	// Determine the thread geometry for the kernel

	dataAnalysis.Analyze(function, inputObjects);
	shapeAnalysis.Analyze(function, inputShapes);

	Analysis::KernelOptionsAnalysis optionsAnalysis;
	optionsAnalysis.Analyze(function, shapeAnalysis);

	auto runtimeOptions = optionsAnalysis.GetInputOptions();

	// Compute the number of input arguments: (function arguments * 2) + return arguments + [return size arguments]

	unsigned int kernelArgumentCount = (function->GetParameterCount() * 2) + function->GetReturnCount();

	// Dynamically sized kernel output parameters (we use this to indicate size of output buffers)

	for (const auto& shape : inputOptions->ReturnWriteShapes)
	{
		// Any return argument that is not determined by the geometry or statically specified
		// needs a kernel argument for outputting the real size

		if (RuntimeUtils::IsDynamicReturnShape(shape, inputOptions->ThreadGeometry))
		{
			kernelArgumentCount++;
		}
	}

	// Dynamically specify the thread geometry (vector: length; list: cells + cell lengths)

	if (const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(inputOptions->ThreadGeometry))
	{
		// [vector size]

		if (Analysis::ShapeUtils::IsDynamicSize(vectorGeometry->GetSize()))
		{
			kernelArgumentCount++;
		}
	}
	else if (Analysis::ShapeUtils::IsShape<Analysis::ListShape>(inputOptions->ThreadGeometry))
	{
		// [Cell threads] + list size + cell sizes

		if (inputOptions->ListCellThreads == Codegen::InputOptions::DynamicSize)
		{
			kernelArgumentCount++;
		}
		kernelArgumentCount += 2;
	}

	// Fetch the handle to the GPU entry function

	auto kernel = program->GetKernel(kernelName, kernelArgumentCount);

	// Determine any data copies that occur

	Analysis::DataCopyAnalysis copyAnalysis(dataAnalysis);
	copyAnalysis.Analyze(function);

	// Execute the compiled kernel on the GPU
	//   1. Create the invocation (thread sizes + arguments)
	//   2. Initialize the arguments
	//   3. Execute
	//   4. Initialize return values

	// Initialize kernel invocation and configure the runtime thread layout

	CUDA::KernelInvocation invocation(kernel);

	const auto [blockSize, blockCount] = GetBlockShape(*runtimeOptions, kernelOptions);
	invocation.SetBlockShape(blockSize, 1, 1);
	invocation.SetGridShape(blockCount, 1, 1);

	// Initialize the input buffers for the kernel

	std::vector<CUDA::Buffer *> inputSizeBuffers;
	for (auto i = 0u; i < function->GetParameterCount(); ++i)
	{
		const auto parameter = function->GetParameter(i);
		const auto argument = arguments.at(i);

		Utils::Logger::LogInfo("Initializing input argument: " + parameter->GetName() + " [" + argument->Description() + "]");

		// Transfer the buffer to the GPU, for input parameters we assume read only

		auto buffer = argument->GetGPUReadBuffer();
		invocation.AddParameter(*buffer);

		// Allocate a size parameter for all inputs

		const auto runtimeShape = runtimeOptions->ParameterShapes.at(parameter);
		inputSizeBuffers.push_back(AllocateSizeBuffer(invocation, runtimeShape, false));
	}

	// Initialize the return buffers for the kernel

	std::vector<DataBuffer *> returnBuffers;
	std::vector<CUDA::Buffer *> returnSizeBuffers;

	for (auto i = 0u; i < function->GetReturnCount(); ++i)
	{
		// Create a new buffer for the return value

		const auto type = function->GetReturnType(i);
		const auto shape = runtimeOptions->ReturnShapes.at(i);

		auto returnBuffer = DataBuffer::CreateEmpty(type, shape);
		returnBuffers.push_back(returnBuffer);

		Utils::Logger::LogInfo("Initializing return argument: " + std::to_string(i) + " [" + returnBuffer->Description() + "]");

		// Transfer the write buffer to te GPU, we assume all returns write (or else...)

		auto gpuBuffer = returnBuffer->GetGPUWriteBuffer();
		gpuBuffer->Clear();
		invocation.AddParameter(*gpuBuffer);

		// Allocate a size parameter if neded

		const auto inputShape = inputOptions->ReturnWriteShapes.at(i);
		if (RuntimeUtils::IsDynamicReturnShape(inputShape, inputOptions->ThreadGeometry))
		{
			returnSizeBuffers.push_back(AllocateSizeBuffer(invocation, shape, true));
		}

		// Copy data if needed from input

		const auto returnObject = dataAnalysis.GetReturnObject(i);
		if (copyAnalysis.ContainsDataCopy(returnObject))
		{
			const auto inputObject = copyAnalysis.GetDataCopy(returnObject);
			auto inputBuffer = inputObject->GetDataBuffer();

			Utils::Logger::LogInfo("Initializing return data: " + std::to_string(i) + " = " + inputObject->ToString() + " -> " + returnObject->ToString());

			CUDA::Buffer::Copy(returnBuffer->GetGPUWriteBuffer(), inputBuffer->GetGPUReadBuffer(), inputBuffer->GetGPUBufferSize());
		}
	}

	// Setup constant dynamic size parameters and allocate the dynamic shared memory according to the kernel

	if (const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(runtimeOptions->ThreadGeometry))
	{
		const auto inputGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(inputOptions->ThreadGeometry);
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
	else if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(runtimeOptions->ThreadGeometry))
	{
		// Add the dynamic thread count to the parameters if needed

		if (inputOptions->ListCellThreads == Codegen::InputOptions::DynamicSize)
		{
			AllocateConstantParameter(invocation, runtimeOptions->ListCellThreads, "<geometry cell threads>");
		}

		// Allocate a buffer with the cell sizes for the execution geometry

		const auto inputListShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(inputOptions->ThreadGeometry);
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

	// Resize return buffers for dynamically sized outputs (compression)

	std::vector<DataBuffer *> resizedBuffers;
	for (auto returnIndex = 0u, resizeIndex = 0u; returnIndex < function->GetReturnCount(); ++returnIndex)
	{
		// Check if the return buffer was a dynamic allocation

		auto returnBuffer = returnBuffers.at(returnIndex);

		const auto inputShape = inputOptions->ReturnShapes.at(returnIndex);
		const auto inputWriteShape = inputOptions->ReturnWriteShapes.at(returnIndex);

		if (RuntimeUtils::IsDynamicReturnShape(inputWriteShape, inputOptions->ThreadGeometry))
		{
			// Get the right buffer kind and resize

			if (const auto vectorBuffer = BufferUtils::GetBuffer<VectorBuffer>(returnBuffer))
			{
				// Transfer the size to the CPU to resize the buffer

				auto size = 0u;
				auto sizeBuffer = returnSizeBuffers.at(resizeIndex++);
				sizeBuffer->SetCPUBuffer(&size);
				sizeBuffer->TransferToCPU();

				// Allocate a new buffer and transfer the contents

				auto resizedBuffer = VectorBuffer::CreateEmpty(vectorBuffer->GetType(), new Analysis::Shape::ConstantSize(size));
				CUDA::Buffer::Copy(resizedBuffer->GetGPUWriteBuffer(), vectorBuffer->GetGPUReadBuffer(), resizedBuffer->GetGPUBufferSize());

				Utils::Logger::LogInfo("Resized dynamic buffer [" + returnBuffer->Description() + "] to [" + resizedBuffer->Description() + "]");

				resizedBuffers.push_back(resizedBuffer);
				delete returnBuffer;
			}
			else
			{
				//TODO: Resize list buffers
				Utils::Logger::LogError("Unable to resize dynamically sized return buffer " + returnBuffer->Description());
			}
		}
		else
		{
			resizedBuffers.push_back(returnBuffer);
		}
	}

	// Deallocate dynamic size CUDA buffers

	for (auto buffer : inputSizeBuffers)
	{
		delete buffer;
	}
	for (auto buffer : returnSizeBuffers)
	{
		delete buffer;
	}

	return resizedBuffers;
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

std::pair<unsigned int, unsigned int> GPUExecutionEngine::GetBlockShape(Codegen::InputOptions& runtimeOptions, const PTX::FunctionOptions& kernelOptions) const
{
	// Compute the block size and count based on the kernel, input and target configurations
	// We assume that all sizes are known at this point

	auto& gpu = m_runtime.GetGPUManager();
	auto& device = gpu.GetCurrentDevice();

	const auto maxBlockSize = device->GetMaxThreadsDimension(0);

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

				if (size < maxBlockSize)
				{
					blockSize = size;
				}
				else
				{
					blockSize = maxBlockSize;
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

			auto blockSize = maxBlockSize;
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
