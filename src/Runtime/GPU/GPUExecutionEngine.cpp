#include "Runtime/GPU/GPUExecutionEngine.h"

#include "CUDA/Buffer.h"
#include "CUDA/Constant.h"
#include "CUDA/Kernel.h"
#include "CUDA/KernelInvocation.h"
#include "CUDA/Utils.h"

#include "Analysis/DataObject/DataObjectAnalysis.h"
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
	
	auto timeKernelInit_start = Utils::Chrono::Start("Kernel '" + function->GetName() + "' initialization");

	const auto program = m_runtime.GetGPUManager().GetProgram();
	const auto kernelName = function->GetName();

	const auto& kernelOptions = program->GetKernelOptions(kernelName);
	const auto inputOptions = kernelOptions.GetCodegenOptions();

	if (m_optionsCache.find(function) == m_optionsCache.end())
	{
		auto timeAnalysis_start = Utils::Chrono::Start("Runtime analysis");

		auto timeAnalysisInit_start = Utils::Chrono::Start("Analysis initialziation");

		// Collect runtime shape information for determining exact thread geometry and return shapes

		Analysis::DataObjectAnalysis dataAnalysis(m_program);
		Analysis::ShapeAnalysis shapeAnalysis(dataAnalysis, m_program, true);

		Analysis::DataObjectAnalysis::Properties inputObjects;
		Analysis::ShapeAnalysis::Properties inputShapes;

		for (auto i = 0u; i < function->GetParameterCount(); ++i)
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

		Utils::Chrono::End(timeAnalysisInit_start);

		// Determine the thread geometry for the kernel

		dataAnalysis.Analyze(function, inputObjects);
		shapeAnalysis.Analyze(function, inputShapes);

		Analysis::KernelOptionsAnalysis optionsAnalysis(shapeAnalysis);
		optionsAnalysis.Analyze(function);

		m_optionsCache[function] = std::move(optionsAnalysis.GetInputOptions());

		Utils::Chrono::End(timeAnalysis_start);
	}
	else
	{
		Utils::Logger::LogDebug("Using cached input options for kernel '" + function->GetName() + "'");
	}

	auto runtimeOptions = m_optionsCache.at(function);

	// Execute the compiled kernel on the GPU
	//   1. Create the invocation (thread sizes + arguments)
	//   2. Initialize the arguments
	//   3. Execute
	//   4. Resize return values

	//TODO: Timing bracket sometimes negative overhead
	auto timeInvocationInit_start = Utils::Chrono::Start("Invocation initialization");

	// Fetch the handle to the GPU entry function and create the invocation

	auto kernel = program->GetKernel(kernelName);
	CUDA::KernelInvocation invocation(kernel);

	// Configure the runtime thread layout

	const auto [blockSize, blockCount] = GetBlockShape(runtimeOptions, kernelOptions);
	invocation.SetBlockShape(blockSize, 1, 1);
	invocation.SetGridShape(blockCount, 1, 1);

	// Initialize the input buffers for the kernel

	std::vector<CUDA::Buffer *> inputSizeBuffers;
	for (auto i = 0u; i < function->GetParameterCount(); ++i)
	{
		const auto parameter = function->GetParameter(i);
		const auto argument = arguments.at(i);

		Utils::Logger::LogDebug("Initializing input argument: " + parameter->GetName() + " [" + argument->Description() + "]");

		// Transfer the buffer to the GPU, for input parameters we assume read only

		auto buffer = argument->GetGPUReadBuffer();
		invocation.AddParameter(*buffer);

		// Allocate a size parameter for all inputs

		const auto runtimeShape = runtimeOptions->ParameterShapes.at(parameter);
		inputSizeBuffers.push_back(AllocateSizeParameter(invocation, runtimeShape, false));
	}

	// Initialize the return buffers for the kernel

	std::vector<DataBuffer *> returnBuffers;
	std::vector<CUDA::Buffer *> returnSizeBuffers;

	if (function->GetReturnCount() > 0)
	{
		// Determine any data copies that occur

		for (auto i = 0u; i < function->GetReturnCount(); ++i)
		{
			// Create a new buffer for the return value

			const auto type = function->GetReturnType(i);
			const auto shape = runtimeOptions->ReturnShapes.at(i);

			auto returnBuffer = DataBuffer::CreateEmpty(type, shape);
			returnBuffers.push_back(returnBuffer);

			Utils::Logger::LogDebug("Initializing return argument: " + std::to_string(i) + " [" + returnBuffer->Description() + "]");

			// Transfer the write buffer to te GPU, we assume all returns write (or else...)

			auto gpuBuffer = returnBuffer->GetGPUWriteBuffer();
			invocation.AddParameter(*gpuBuffer);

			// Allocate a size parameter if neded

			const auto returnShape = inputOptions->ReturnShapes.at(i);
			const auto returnWriteShape = inputOptions->ReturnWriteShapes.at(i);
			if (RuntimeUtils::IsDynamicReturnShape(returnShape, returnWriteShape, inputOptions->ThreadGeometry))
			{
				returnSizeBuffers.push_back(AllocateSizeParameter(invocation, shape, true));
			}

			// Copy data if needed from input

			const auto& dataCopies = runtimeOptions->CopyObjects;
			const auto returnObject = runtimeOptions->ReturnObjects.at(i);

			if (dataCopies.find(returnObject) != dataCopies.end())
			{
				const auto inputObject = dataCopies.at(returnObject);
				auto inputBuffer = inputObject->GetDataBuffer();

				Utils::Logger::LogDebug("Initializing return data: " + std::to_string(i) + " = " + inputObject->ToString() + " -> " + returnObject->ToString());

				if (Analysis::ShapeUtils::IsShape<Analysis::VectorShape>(shape))
				{
					auto vectorInput = BufferUtils::GetBuffer<VectorBuffer>(inputBuffer);
					auto vectorReturn = BufferUtils::GetBuffer<VectorBuffer>(returnBuffer);

					CUDA::Buffer::Copy(vectorReturn->GetGPUWriteBuffer(), vectorInput->GetGPUReadBuffer(), vectorInput->GetGPUBufferSize());
				}
				else if (Analysis::ShapeUtils::IsShape<Analysis::ListShape>(shape))
				{
					//TODO: Initialize list return buffer
				}
				else
				{
					Utils::Logger::LogError("Unable to initalize return buffer shape " + Analysis::ShapeUtils::ShapeString(shape));
				}
			}
			else
			{
				returnBuffer->Clear();
			}
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

		AllocateListSizeParameter(invocation, listShape, "<geometry cell sizes>");

		//TODO: Determine the correct amount of shared memory for cells
		invocation.SetDynamicSharedMemorySize(kernelOptions.GetDynamicSharedMemorySize() * 2);
	}

	// Load extra parameters for the kernel

	for (auto i = function->GetParameterCount(); i < arguments.size(); ++i)
	{
		const auto argument = arguments.at(i);

		Utils::Logger::LogDebug("Initializing extra input argument: " + std::to_string(i) + " [" + argument->Description() + "]");

		// Transfer the buffer to the GPU, for input parameters we assume read only. Extra parameters have no shape

		auto buffer = argument->GetGPUReadBuffer();
		invocation.AddParameter(*buffer);
	}

	Utils::Chrono::End(timeInvocationInit_start);
	Utils::Chrono::End(timeKernelInit_start);

	// Launch the kernel!

	invocation.Launch();

	CUDA::Synchronize();

	// Deallocate input size buffers

	for (auto buffer : inputSizeBuffers)
	{
		delete buffer;
	}

	// Resize return buffers for dynamically sized outputs (compression)

	if (returnSizeBuffers.size() > 0)
	{
		auto timeResize_start = Utils::Chrono::Start("Resize buffers");

		std::vector<DataBuffer *> resizedBuffers;
		for (auto returnIndex = 0u, resizeIndex = 0u; returnIndex < function->GetReturnCount(); ++returnIndex)
		{
			// Check if the return buffer was a dynamic allocation

			auto returnBuffer = returnBuffers.at(returnIndex);

			const auto returnShape = inputOptions->ReturnShapes.at(returnIndex);
			const auto returnWriteShape = inputOptions->ReturnWriteShapes.at(returnIndex);
			
			if (RuntimeUtils::IsDynamicReturnShape(returnShape, returnWriteShape, inputOptions->ThreadGeometry))
			{
				// Resize the buffer according to the dynamic size

				auto sizeBuffer = returnSizeBuffers.at(resizeIndex++);
				auto resizedBuffer = ResizeBuffer(returnBuffer, sizeBuffer);

				resizedBuffers.push_back(resizedBuffer);
			}
			else
			{
				resizedBuffers.push_back(returnBuffer);
			}
		}

		// Deallocate return size CUDA buffers

		for (auto buffer : returnSizeBuffers)
		{
			delete buffer;
		}

		Utils::Chrono::End(timeResize_start);

		return resizedBuffers;
	}

	return {returnBuffers};
}

VectorBuffer *GPUExecutionEngine::ResizeBuffer(VectorBuffer *vectorBuffer, std::uint32_t size) const
{
	// Check if resize necessary

	if (vectorBuffer->GetElementCount() != size)
	{
		// Allocate a new buffer and transfer the contents if substantially smaller than the allocated size

		auto resizedBuffer = VectorBuffer::CreateEmpty(vectorBuffer->GetType(), new Analysis::Shape::ConstantSize(size));
		auto gpuBuffer = vectorBuffer->GetGPUReadBuffer();

		if (resizedBuffer->GetGPUBufferSize() < gpuBuffer->GetAllocatedSize() * 0.9)
		{
			// Copy data to new buffer

			CUDA::Buffer::Copy(resizedBuffer->GetGPUWriteBuffer(), gpuBuffer, resizedBuffer->GetGPUBufferSize());
		}
		else
		{
			// Move the buffer from one VectorBuffer to another

			gpuBuffer->SetCPUBuffer(nullptr);
			gpuBuffer->SetSize(size);
			resizedBuffer->SetGPUBuffer(gpuBuffer);

			vectorBuffer->SetGPUBuffer(nullptr);
			vectorBuffer->InvalidateGPU();
		}

		Utils::Logger::LogDebug("Resized vector buffer [" + vectorBuffer->Description() + "] to [" + resizedBuffer->Description() + "]");

		delete vectorBuffer;
		return resizedBuffer;
	}
	return vectorBuffer;
}

ListBuffer *GPUExecutionEngine::ResizeBuffer(ListBuffer *listBuffer, const std::vector<std::uint32_t>& sizes) const
{
	// For each cell, allocate a new buffer and transfer the contents

	std::vector<DataBuffer *> resizedCells;
	for (auto i = 0u; i < listBuffer->GetCellCount(); ++i)
	{
		const auto size = sizes.at((sizes.size() == 1) ? 0 : i);
		const auto cellBuffer = BufferUtils::GetBuffer<VectorBuffer>(listBuffer->GetCell(i));

		resizedCells.push_back(ResizeBuffer(cellBuffer, size));
	}

	auto resizedBuffer = new ListBuffer(resizedCells);

	Utils::Logger::LogDebug("Resized list buffer [" + listBuffer->Description() + "] to [" + resizedBuffer->Description() + "]");

	delete listBuffer;
	return resizedBuffer;
}

DataBuffer *GPUExecutionEngine::ResizeBuffer(DataBuffer *dataBuffer, CUDA::Buffer *sizeBuffer) const
{
	// Transfer the size to the CPU to resize the buffer

	if (const auto vectorBuffer = BufferUtils::GetBuffer<VectorBuffer>(dataBuffer, false))
	{
		auto size = 0u;
		sizeBuffer->SetCPUBuffer(&size);
		sizeBuffer->TransferToCPU();
		return ResizeBuffer(vectorBuffer, size);
	}
	else if (const auto listBuffer = BufferUtils::GetBuffer<ListBuffer>(dataBuffer, false))
	{
		std::vector<std::uint32_t> cellSizes(listBuffer->GetCellCount());
		sizeBuffer->SetCPUBuffer(cellSizes.data());
		sizeBuffer->TransferToCPU();
		return ResizeBuffer(listBuffer, cellSizes);
	}
	else
	{
		Utils::Logger::LogError("Unable to resize data buffer " + dataBuffer->Description());
	}
}

std::pair<unsigned int, unsigned int> GPUExecutionEngine::GetBlockShape(Codegen::InputOptions *runtimeOptions, const PTX::FunctionOptions& kernelOptions) const
{
	// Compute the block size and count based on the kernel, input and target configurations
	// We assume that all sizes are known at this point

	auto& gpu = m_runtime.GetGPUManager();
	auto& device = gpu.GetCurrentDevice();

	const auto maxBlockSize = device->GetMaxThreadsDimension(0);

	const auto threadGeometry = runtimeOptions->ThreadGeometry;
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

				cellSize = runtimeOptions->ListCellThreads;

				if (kernelOptions.GetThreadMultiple() != 0)
				{
					// Ensure the thread number is a multiple of the kernel specification

					auto multiple = kernelOptions.GetThreadMultiple();
					cellSize = ((cellSize + multiple - 1) / multiple) * multiple;
				}
			}

			// Important: Update the number of cells per thread

			runtimeOptions->ListCellThreads = cellSize;

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

template<typename T>
void GPUExecutionEngine::AllocateConstantParameter(CUDA::KernelInvocation& invocation, const T& value, const std::string& description) const
{
	Utils::Logger::LogDebug("Initializing constant input argument: " + description + " [" + std::to_string(sizeof(T)) + " bytes = " + std::to_string(value) + "]");

	auto sizeConstant = new CUDA::TypedConstant<T>(value);
	invocation.AddParameter(*sizeConstant);
}

CUDA::Buffer *GPUExecutionEngine::AllocateListSizeParameter(CUDA::KernelInvocation& invocation, const Analysis::ListShape *shape, const std::string& description) const
{
	// Form a vector of cell sizes, for lists of constan-sized vectors

	std::vector<std::uint32_t> cellSizes;
	if (const auto constantSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(shape->GetListSize()))
	{
		const auto& elementShapes = shape->GetElementShapes();
		auto cellCount = constantSize->GetValue();

		if (elementShapes.size() == 1 && cellCount > 1)
		{
			if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(elementShapes.at(0)))
			{
				if (const auto constantSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(vectorShape->GetSize()))
				{
					for (auto i = 0u; i < cellCount; ++i)
					{
						cellSizes.push_back(constantSize->GetValue());
					}
				}
				else
				{
					Utils::Logger::LogError("Unable to get constant cell size for list shape " + Analysis::ShapeUtils::ShapeString(shape));
				}
			}
		}
		else if (elementShapes.size() == cellCount)
		{
			for (const auto cellShape : shape->GetElementShapes())
			{
				if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellShape))
				{
					if (const auto constantSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(vectorShape->GetSize()))
					{
						cellSizes.push_back(constantSize->GetValue());
					}
					else
					{
						Utils::Logger::LogError("Unable to get constant cell size for list shape " + Analysis::ShapeUtils::ShapeString(shape));
					}
				}
			}
		}
		else
		{
			Utils::Logger::LogError("Mismatched cell count and list size for shape " + Analysis::ShapeUtils::ShapeString(shape));
		}
	}
	else
	{
		Utils::Logger::LogError("Unable to get constant cell count for list shape " + Analysis::ShapeUtils::ShapeString(shape));
	}

	Utils::Logger::LogDebug("Initializing size argument: " + description + " [u32(" + std::to_string(sizeof(std::uint32_t)) + " bytes) x " + std::to_string(cellSizes.size()) + "]");

	// Transfer to the GPU

	auto buffer = new CUDA::Buffer(cellSizes.data(), cellSizes.size() * sizeof(std::uint32_t));
	buffer->AllocateOnGPU();
	buffer->TransferToGPU();

	//TODO: Save the size buffer on the list object to re-use

	invocation.AddParameter(*buffer);
	return buffer;
}

CUDA::Buffer *GPUExecutionEngine::AllocateSizeParameter(CUDA::KernelInvocation& invocation, const Analysis::Shape *shape, bool returnParameter) const
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

		Utils::Logger::LogDebug("Initializing size argument: <dynamic vector size> [u32(" + std::to_string(sizeof(std::uint32_t)) + " bytes) x 1]");

		auto buffer = new CUDA::Buffer(sizeof(std::uint32_t));
		buffer->AllocateOnGPU();
		buffer->Clear();
		invocation.AddParameter(*buffer);

		return buffer;
	}
	else if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
	{
		// A list size is given by the cell sizes (the number of cells is specified through another parameter)

		return AllocateListSizeParameter(invocation, listShape, "<dynamic cell sizes>");
	}
	else
	{
		Utils::Logger::LogError("Unable to allocate size buffer for " + Analysis::ShapeUtils::ShapeString(shape) + "[unsupported shape]");
	}
}

}
