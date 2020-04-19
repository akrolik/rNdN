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

	for (auto i = 0u; i < function->GetParameterCount(); ++i)
	{
		const auto parameter = function->GetParameter(i);
		const auto argument = arguments.at(i);

		Utils::Logger::LogDebug("Initializing input argument: " + parameter->GetName() + " [" + argument->Description() + "]");

		// Transfer the buffer to the GPU, for input parameters we assume read only

		auto buffer = argument->GetGPUReadBuffer();
		invocation.AddParameter(*buffer);

		// Add a size parameter for each input

		auto sizeBuffer = argument->GetGPUSizeBuffer();
		invocation.AddParameter(*sizeBuffer);
	}

	// Initialize the return buffers for the kernel

	std::vector<DataBuffer *> returnBuffers;
	if (function->GetReturnCount() > 0)
	{
		// Determine any data copies that occur

		for (auto i = 0u; i < function->GetReturnCount(); ++i)
		{
			const auto& dataCopies = runtimeOptions->CopyObjects;
			const auto returnObject = runtimeOptions->ReturnObjects.at(i);

			DataBuffer *returnBuffer = nullptr;
			if (dataCopies.find(returnObject) != dataCopies.end())
			{
				// Create a new buffer as a copy of an input object

				const auto inputObject = dataCopies.at(returnObject);
				const auto inputBuffer = inputObject->GetDataBuffer();

				returnBuffer = inputBuffer->Clone();

				Utils::Logger::LogDebug(
					"Initializing return argument: " + std::to_string(i) + " = " + inputObject->ToString() + " -> " + returnObject->ToString() +
					"[" + returnBuffer->Description() + "]"
				);
			}
			else
			{
				// Create a new buffer for the return value

				const auto type = function->GetReturnType(i);
				const auto shape = runtimeOptions->ReturnShapes.at(i);

				returnBuffer = DataBuffer::CreateEmpty(type, shape);
				returnBuffer->Clear();

				Utils::Logger::LogDebug("Initializing return argument: " + std::to_string(i) + " [" + returnBuffer->Description() + "]");
			}
			returnBuffers.push_back(returnBuffer);

			// Transfer the write buffer to the GPU, we assume all returns write (or else...)

			auto gpuBuffer = returnBuffer->GetGPUWriteBuffer();
			invocation.AddParameter(*gpuBuffer);

			// Add a dynamic size parameter if neded

			const auto returnShape = inputOptions->ReturnShapes.at(i);
			const auto returnWriteShape = inputOptions->ReturnWriteShapes.at(i);
			if (RuntimeUtils::IsDynamicReturnShape(returnShape, returnWriteShape, inputOptions->ThreadGeometry))
			{
				// Clear the size buffer for the dynamic output data

				auto sizeBuffer = returnBuffer->GetGPUSizeBuffer();
				sizeBuffer->Clear();

				invocation.AddParameter(*sizeBuffer);
			}

		}
	}

	// Setup constant dynamic size parameters and allocate the dynamic shared memory according to the kernel

	std::vector<CUDA::Data *> dynamicBuffers;
	std::vector<std::uint32_t> dynamicCellSizes;

	if (const auto runtimeVectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(runtimeOptions->ThreadGeometry))
	{
		const auto inputVectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(inputOptions->ThreadGeometry);
		if (Analysis::ShapeUtils::IsDynamicSize(inputVectorGeometry->GetSize()))
		{
			if (const auto constantSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(runtimeVectorGeometry->GetSize()))
			{
				dynamicBuffers.push_back(AllocateConstantParameter(invocation, constantSize->GetValue(), "<vector geometry>"));
			}
			else
			{
				Utils::Logger::LogError("Invocation thread geometry must be constant vector " + Analysis::ShapeUtils::ShapeString(runtimeVectorGeometry));
			}
		}

		invocation.SetDynamicSharedMemorySize(kernelOptions.GetDynamicSharedMemorySize());
	}
	else if (const auto runtimeListGeometry = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(runtimeOptions->ThreadGeometry))
	{
		// Add the dynamic thread count to the parameters if needed

		if (inputOptions->ListCellThreads == Codegen::InputOptions::DynamicSize)
		{
			dynamicBuffers.push_back(AllocateConstantParameter(invocation, runtimeOptions->ListCellThreads, "<list geometry threads>"));
		}

		// Allocate a buffer with the cell sizes for the execution geometry

		const auto inputListGeometry = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(inputOptions->ThreadGeometry);
		if (const auto listSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(runtimeListGeometry->GetListSize()))
		{
			auto cellCount = listSize->GetValue();

			// Number of cells parameter if dynamic

			if (Analysis::ShapeUtils::IsDynamicSize(inputListGeometry->GetListSize()))
			{
				dynamicBuffers.push_back(AllocateConstantParameter(invocation, cellCount, "<list geometry size>"));
			}

			// Form a vector of cell sizes, for lists of constan-sized vectors

			const auto& cellShapes = runtimeListGeometry->GetElementShapes();
			for (const auto cellShape : cellShapes)
			{
				if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellShape))
				{
					if (const auto cellSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(vectorShape->GetSize()))
					{
						if (cellShapes.size() == 1 && cellCount > 1)
						{
							for (auto i = 0u; i < cellCount; ++i)
							{
								dynamicCellSizes.push_back(cellSize->GetValue());
							}
						}
						else if (cellShapes.size() == cellCount)
						{
							dynamicCellSizes.push_back(cellSize->GetValue());
						}
						else
						{
							Utils::Logger::LogError("Mismatched cell count and list size for shape " + Analysis::ShapeUtils::ShapeString(runtimeListGeometry));
						}
					}
					else
					{
						Utils::Logger::LogError("Invocation thread geometry must be constant list " + Analysis::ShapeUtils::ShapeString(runtimeListGeometry));
					}
				}
			}

			Utils::Logger::LogDebug("Initializing size argument: <list geometry cell sizes> [u32(4 bytes) x " + std::to_string(dynamicCellSizes.size()) + "]");

			// Transfer to the GPU

			auto buffer = new CUDA::Buffer(dynamicCellSizes.data(), dynamicCellSizes.size() * sizeof(std::uint32_t));
			buffer->AllocateOnGPU();
			buffer->TransferToGPU();
			invocation.AddParameter(*buffer);

			dynamicBuffers.push_back(buffer);
		}
		else
		{
			Utils::Logger::LogError("Invocation thread geometry must be constant list " + Analysis::ShapeUtils::ShapeString(runtimeListGeometry));
		}

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

	Utils::Logger::LogDebug("Kernel '" + function->GetName() + "' complete");

	// Deallocate dynamic buffers

	for (auto buffer : dynamicBuffers)
	{
		delete buffer;
	}

	// Resize return buffers for dynamically sized outputs (compression)

	if (returnBuffers.size() > 0)
	{
		auto timeResize_start = Utils::Chrono::Start("Resize buffers");

		for (auto returnIndex = 0u; returnIndex < function->GetReturnCount(); ++returnIndex)
		{
			// Check if the return buffer was a dynamic allocation

			auto returnBuffer = returnBuffers.at(returnIndex);

			const auto returnShape = inputOptions->ReturnShapes.at(returnIndex);
			const auto returnWriteShape = inputOptions->ReturnWriteShapes.at(returnIndex);
			
			if (RuntimeUtils::IsDynamicReturnShape(returnShape, returnWriteShape, inputOptions->ThreadGeometry))
			{
				// Resize the buffer according to the dynamic size

				returnBuffer->ReallocateGPUBuffer();
			}
		}

		Utils::Chrono::End(timeResize_start);
	}

	return {returnBuffers};
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
CUDA::TypedConstant<T> *GPUExecutionEngine::AllocateConstantParameter(CUDA::KernelInvocation& invocation, const T& value, const std::string& description) const
{
	Utils::Logger::LogDebug("Initializing constant input argument: " + description + " [" + std::to_string(sizeof(T)) + " bytes = " + std::to_string(value) + "]");

	auto sizeConstant = new CUDA::TypedConstant<T>(value);
	invocation.AddParameter(*sizeConstant);

	return sizeConstant;
}

}
