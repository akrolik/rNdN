#include "Runtime/GPU/ExecutionEngine.h"

#include "CUDA/Buffer.h"
#include "CUDA/BufferManager.h"
#include "CUDA/Constant.h"
#include "CUDA/ConstantBuffer.h"
#include "CUDA/Kernel.h"
#include "CUDA/KernelInvocation.h"
#include "CUDA/Utils.h"

#include "HorseIR/Analysis/DataObject/DataObjectAnalysis.h"
#include "HorseIR/Analysis/Geometry/KernelOptionsAnalysis.h"
#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeAnalysis.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

#include "Runtime/RuntimeUtils.h"
#include "Runtime/DataBuffers/BufferUtils.h"

#include "Utils/Logger.h"
#include "Utils/Math.h"

namespace Runtime {
namespace GPU {

std::vector<DataBuffer *> ExecutionEngine::Execute(const HorseIR::Function *function, const std::vector<const DataBuffer *>& arguments)
{
	// Get the input options used for codegen
	
	auto timeKernelInit_start = Utils::Chrono::Start("Kernel '" + function->GetName() + "' initialization");

	const auto program = m_runtime.GetGPUManager().GetProgram();
	const auto kernelName = function->GetName();
	const auto kernelCode = program->GetKernelCode(kernelName);
	const auto inputOptions = kernelCode->GetCodegenOptions();

	if (m_optionsCache.find(function) == m_optionsCache.end())
	{
		auto timeAnalysis_start = Utils::Chrono::Start("Runtime analysis");

		auto timeAnalysisInit_start = Utils::Chrono::Start("Analysis initialziation");

		// Collect runtime shape information for determining exact thread geometry and return shapes

		HorseIR::Analysis::DataObjectAnalysis dataAnalysis(m_program);
		dataAnalysis.SetCollectInSets(false);
		dataAnalysis.SetCollectOutSets(false);

		HorseIR::Analysis::ShapeAnalysis shapeAnalysis(dataAnalysis, m_program, true);
		shapeAnalysis.SetCollectOutSets(false);

		HorseIR::Analysis::DataObjectAnalysis::Properties inputObjects;
		HorseIR::Analysis::ShapeAnalysis::Properties inputShapes;

		for (auto i = 0u; i < function->GetParameterCount(); ++i)
		{
			const auto parameter = function->GetParameter(i);
			const auto symbol = parameter->GetSymbol();

			const auto object = inputOptions->ParameterObjects.at(parameter);
			const auto runtimeObject = new HorseIR::Analysis::DataObject(object->GetObjectID(), arguments.at(i));
			inputObjects[symbol] = runtimeObject;

			// Setup compression constraints

			const auto symbolShape = inputOptions->ParameterShapes.at(parameter);
			const auto runtimeShape = arguments.at(i)->GetShape();

			if (const auto vectorShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(symbolShape))
			{
				if (const auto vectorSize = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::CompressedSize>(vectorShape->GetSize()))
				{
					const auto runtimeVector = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(runtimeShape);
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

		HorseIR::Analysis::KernelOptionsAnalysis optionsAnalysis(shapeAnalysis);
		m_optionsCache[function] = optionsAnalysis.Analyze(function);

		Utils::Chrono::End(timeAnalysis_start);
	}
	else
	{
		if (Utils::Options::IsDebug_Print())
		{
			Utils::Logger::LogDebug("Using cached input options for kernel '" + function->GetName() + "'");
		}
	}

	auto runtimeOptions = m_optionsCache.at(function);

	// Execute the compiled kernel on the GPU
	//   1. Create the invocation (thread sizes + arguments)
	//   2. Initialize the arguments
	//   3. Execute
	//   4. Resize return values

	auto timeInvocationInit_start = Utils::Chrono::Start("Invocation initialization");

	// Fetch the handle to the GPU entry function and create the invocation

	auto kernel = program->GetKernel(kernelName);
	CUDA::KernelInvocation invocation(kernel);

	// Configure the runtime thread layout

	const auto [blockSize, blockCount] = GetBlockShape(runtimeOptions, kernelCode, kernel);
	invocation.SetBlockShape(blockSize, 1, 1);
	invocation.SetGridShape(blockCount, 1, 1);

	// Initialize the input buffers for the kernel

	for (auto i = 0u; i < function->GetParameterCount(); ++i)
	{
		const auto parameter = function->GetParameter(i);
		const auto argument = arguments.at(i);

		if (Utils::Options::IsDebug_Print())
		{
			Utils::Logger::LogDebug("Initializing input argument: " + parameter->GetName() + " [" + argument->Description() + "]");
		}

		// Transfer the buffer to the GPU, for input parameters we assume read only

		auto buffer = argument->GetGPUReadBuffer();
		invocation.AddParameter(*buffer);

		// Add a size parameter for each input

		auto sizeBuffer = argument->GetGPUSizeBuffer();
		invocation.AddParameter(*sizeBuffer);
	}

	// Initialize the return buffers for the kernel

	std::vector<DataBuffer *> returnBuffers;
	for (auto i = 0u; i < function->GetReturnCount(); ++i)
	{
		const auto& dataInit = runtimeOptions->InitObjects;
		const auto& dataCopies = runtimeOptions->CopyObjects;
		const auto returnObject = runtimeOptions->ReturnObjects.at(i);

		const auto type = function->GetReturnType(i);
		const auto shape = runtimeOptions->ReturnShapes.at(i);

		// Determine any data copies that occur

		DataBuffer *returnBuffer = nullptr;
		if (dataInit.find(returnObject) != dataInit.end())
		{
			auto initialization = dataInit.at(returnObject);
			if (initialization != HorseIR::Analysis::DataInitializationAnalysis::Initialization::Copy)
			{
				returnBuffer = DataBuffer::CreateEmpty(type, shape);
				returnBuffer->RequireGPUConsistent(true);
			}

			auto timeInitializeBuffer_start = Utils::Chrono::Start("Initialize buffer");

			std::string description;
			switch (initialization)
			{
				case HorseIR::Analysis::DataInitializationAnalysis::Initialization::Clear:
				{
					returnBuffer->Clear(DataBuffer::ClearMode::Zero);
					description = " = <clear>";
					break;
				}
				case HorseIR::Analysis::DataInitializationAnalysis::Initialization::Minimum:
				{
					returnBuffer->Clear(DataBuffer::ClearMode::Minimum);
					description = " = <min>";
					break;
				}
				case HorseIR::Analysis::DataInitializationAnalysis::Initialization::Maximum:
				{
					returnBuffer->Clear(DataBuffer::ClearMode::Maximum);
					description = " = <max>";
					break;
				}
				case HorseIR::Analysis::DataInitializationAnalysis::Initialization::Copy:
				{
					// Create a new buffer as a copy of an input object

					const auto inputObject = dataCopies.at(returnObject);
					const auto inputBuffer = inputObject->GetDataBuffer();

					returnBuffer = inputBuffer->Clone();
					description = " = <copy> " + inputObject->ToString() + " -> " + returnObject->ToString();
					break;
				}
			}

			if (Utils::Options::IsDebug_Print())
			{
				Utils::Logger::LogDebug("Initializing return argument: " + std::to_string(i) + description + " [" + returnBuffer->Description() + "]");
			}

			Utils::Chrono::End(timeInitializeBuffer_start);
		}
		else
		{
			returnBuffer = DataBuffer::CreateEmpty(type, shape);

			if (Utils::Options::IsDebug_Print())
			{
				Utils::Logger::LogDebug("Initializing return argument: " + std::to_string(i) + " [" + returnBuffer->Description() + "]");
			}

		}

		returnBuffers.push_back(returnBuffer);

		// Transfer the write buffer to the GPU, we assume all returns write (or else...)

		auto gpuBuffer = returnBuffer->GetGPUWriteBuffer();
		invocation.AddParameter(*gpuBuffer);

		// Add a dynamic size parameter if neded

		const auto returnShape = inputOptions->ReturnShapes.at(i);
		const auto returnWriteShape = inputOptions->ReturnWriteShapes.at(i);

		auto sizeBuffer = returnBuffer->GetGPUSizeBuffer();

		if (RuntimeUtils::IsDynamicReturnShape(returnShape, returnWriteShape, inputOptions->ThreadGeometry))
		{
			// Clear the size buffer for the dynamic output data

			auto timeInitializeSize_start = Utils::Chrono::Start("Initialize buffer");
			if (HorseIR::Analysis::ShapeUtils::IsShape<HorseIR::Analysis::VectorShape>(returnShape))
			{
				sizeBuffer->Clear();
			}
			else if (HorseIR::Analysis::ShapeUtils::IsShape<HorseIR::Analysis::ListShape>(returnShape))
			{
				auto listReturnBuffer = BufferUtils::GetBuffer<ListBuffer>(returnBuffer);
				for (auto cellBuffer : listReturnBuffer->GetCells())
				{
					auto cellSizeBuffer = cellBuffer->GetGPUSizeBuffer();
					cellSizeBuffer->Clear();
				}
			}
			Utils::Chrono::End(timeInitializeSize_start);
		}

		invocation.AddParameter(*sizeBuffer);
	}

	// Setup constant dynamic size parameters and allocate the dynamic shared memory according to the kernel

	std::vector<CUDA::Data *> dynamicBuffers;
	if (const auto runtimeVectorGeometry = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(runtimeOptions->ThreadGeometry))
	{
		const auto inputVectorGeometry = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(inputOptions->ThreadGeometry);
		if (HorseIR::Analysis::ShapeUtils::IsDynamicSize(inputVectorGeometry->GetSize()))
		{
			if (const auto constantSize = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(runtimeVectorGeometry->GetSize()))
			{
				dynamicBuffers.push_back(AllocateConstantParameter(invocation, constantSize->GetValue(), "<vector geometry>"));
			}
			else
			{
				Utils::Logger::LogError("Invocation thread geometry must be constant vector " + HorseIR::Analysis::ShapeUtils::ShapeString(runtimeVectorGeometry));
			}
		}

		invocation.SetDynamicSharedMemorySize(kernelCode->GetDynamicSharedMemorySize());
	}
	else if (const auto runtimeListGeometry = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(runtimeOptions->ThreadGeometry))
	{
		// Add the dynamic thread count to the parameters if needed

		if (inputOptions->ListCellThreads == Frontend::Codegen::InputOptions::DynamicSize)
		{
			dynamicBuffers.push_back(AllocateConstantParameter(invocation, runtimeOptions->ListCellThreads, "<list geometry threads>"));
		}

		// Allocate a buffer with the cell sizes for the execution geometry

		const auto inputListGeometry = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(inputOptions->ThreadGeometry);
		if (const auto listSize = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(runtimeListGeometry->GetListSize()))
		{
			// Number of cells parameter if dynamic

			auto cellCount = listSize->GetValue();
			if (HorseIR::Analysis::ShapeUtils::IsDynamicSize(inputListGeometry->GetListSize()))
			{
				dynamicBuffers.push_back(AllocateConstantParameter(invocation, cellCount, "<list geometry size>"));
			}

			// Form a vector of cell sizes, for lists of constan-sized vectors

			const auto& cellShapes = runtimeListGeometry->GetElementShapes();

			CUDA::Vector<std::uint32_t> dynamicCellSizes;
			for (const auto cellShape : cellShapes)
			{
				if (const auto vectorShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(cellShape))
				{
					auto cellSize = vectorShape->GetSize();
					if (const auto constantSize = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(cellSize))
					{
						if (cellShapes.size() == 1 && cellCount > 1)
						{
							for (auto i = 0u; i < cellCount; ++i)
							{
								dynamicCellSizes.push_back(constantSize->GetValue());
							}
						}
						else if (cellShapes.size() == cellCount)
						{
							dynamicCellSizes.push_back(constantSize->GetValue());
						}
						else
						{
							Utils::Logger::LogError("Mismatched cell count and list size for shape " + HorseIR::Analysis::ShapeUtils::ShapeString(runtimeListGeometry));
						}
					}
					else if (const auto rangedSize = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::RangedSize>(cellSize))
					{
						dynamicBuffers.push_back(AllocateConstantVectorParameter(invocation, rangedSize->GetValues(), "<list geometry cell sizes>"));
					}
					else
					{
						Utils::Logger::LogError("Invocation thread geometry must be constant list " + HorseIR::Analysis::ShapeUtils::ShapeString(runtimeListGeometry));
					}
				}
			}

			if (dynamicCellSizes.size() > 0)
			{
				dynamicBuffers.push_back(AllocateConstantVectorParameter(invocation, dynamicCellSizes, "<list geometry cell sizes>"));
			}
		}
		else
		{
			Utils::Logger::LogError("Invocation thread geometry must be constant list " + HorseIR::Analysis::ShapeUtils::ShapeString(runtimeListGeometry));
		}

		auto blockCells = blockSize / runtimeOptions->ListCellThreads;
		invocation.SetDynamicSharedMemorySize(kernelCode->GetDynamicSharedMemorySize() * blockCells);
	}

	// Load extra parameters for the kernel

	for (auto i = function->GetParameterCount(); i < arguments.size(); ++i)
	{
		const auto argument = arguments.at(i);

		if (Utils::Options::IsDebug_Print())
		{
			Utils::Logger::LogDebug("Initializing extra input argument: " + std::to_string(i) + " [" + argument->Description() + "]");
		}

		// Transfer the buffer to the GPU, for input parameters we assume read only. Extra parameters have no shape

		auto buffer = argument->GetGPUReadBuffer();
		invocation.AddParameter(*buffer);
	}

	CUDA::Synchronize();

	Utils::Chrono::End(timeInvocationInit_start);
	Utils::Chrono::End(timeKernelInit_start);

	// Launch the kernel!

	auto timeKernel_start = Utils::Chrono::Start("Kernel '" + function->GetName() + "' execution");

	invocation.Launch();
	CUDA::Synchronize();

	Utils::Chrono::End(timeKernel_start);

	if (Utils::Options::IsDebug_Print())
	{
		Utils::Logger::LogDebug("Kernel '" + function->GetName() + "' complete");
	}

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

std::pair<unsigned int, unsigned int> ExecutionEngine::GetBlockShape(Frontend::Codegen::InputOptions *runtimeOptions, const PTX::FunctionDefinition<PTX::VoidType> *kernelCode, const CUDA::Kernel& kernel) const
{
	// Compute the block size and count based on the kernel, input and target configurations
	// We assume that all sizes are known at this point

	const auto maxBlockSize = kernel.GetMaxThreads();

	const auto threadGeometry = runtimeOptions->ThreadGeometry;
	if (const auto vectorGeometry = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(threadGeometry))
	{
		if (const auto constantSize = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(vectorGeometry->GetSize()))
		{
			auto size = constantSize->GetValue();
			if (size == 0)
			{
				Utils::Logger::LogError("Zero size kernel for thread geometry " + HorseIR::Analysis::ShapeUtils::ShapeString(threadGeometry));
			}

			auto blockSize = std::get<0>(kernelCode->GetRequiredThreads());
			if (blockSize == 0)
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

				// Maximize the block size based on the GPU and thread multiple

				auto multiple = std::get<0>(kernelCode->GetThreadMultiples());
				if (multiple != 0)
				{
					blockSize = Utils::Math::RoundUp(blockSize, multiple);
				}

				// Power 2 number of threads (e.g order)

				if (kernelCode->GetThreadsPower2())
				{
					blockSize = Utils::Math::Power2Floor(blockSize);
				}
			}
			else
			{
				if (blockSize > maxBlockSize)
				{
					Utils::Logger::LogError("Block size exceeds maximum [" + std::to_string(blockSize) + " > " + std::to_string(maxBlockSize) + "]");
				}
			}

			auto blockCount = ((size + blockSize - 1) / blockSize);
			return {blockSize, blockCount};
		}
	}
	else if (const auto listGeometry = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(threadGeometry))
	{
		if (const auto constantSize = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(listGeometry->GetListSize()))
		{
			auto cellCount = constantSize->GetValue();
			if (cellCount == 0)
			{
				Utils::Logger::LogError("Zero size kernel for thread geometry " + HorseIR::Analysis::ShapeUtils::ShapeString(threadGeometry));
			}

			auto cellSize = std::get<0>(kernelCode->GetRequiredThreads());

			// Check if the cell size is specified as constant or dynamic

			if (cellSize == 0)
			{
				// The thread number was not specified in the input or kernel properties, but determined
				// at runtime depending on the cell sizes. Default to max

				cellSize = runtimeOptions->ListCellThreads;
				if (cellSize == 0)
				{
					cellSize = maxBlockSize;
				}

				// Ensure the thread number is a multiple of the kernel specification

				auto multiple = std::get<0>(kernelCode->GetThreadMultiples());
				if (multiple != 0)
				{
					cellSize = Utils::Math::RoundUp(cellSize, multiple);
				}

				// Power 2 number of threads (e.g order)

				if (kernelCode->GetThreadsPower2())
				{
					cellSize = Utils::Math::Power2Floor(cellSize);
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

	Utils::Logger::LogError("Unknown block shape for thread geometry " + HorseIR::Analysis::ShapeUtils::ShapeString(threadGeometry));
}

template<typename T>
CUDA::TypedConstant<T> *ExecutionEngine::AllocateConstantParameter(CUDA::KernelInvocation& invocation, const T& value, const std::string& description) const
{
	if (Utils::Options::IsDebug_Print())
	{
		Utils::Logger::LogDebug("Initializing constant input argument: " + description + " [" + std::to_string(sizeof(T)) + " bytes = " + std::to_string(value) + "]");
	}

	auto sizeConstant = new CUDA::TypedConstant<T>(value);
	invocation.AddParameter(*sizeConstant);

	return sizeConstant;
}

template<typename T>
CUDA::ConstantBuffer *ExecutionEngine::AllocateConstantVectorParameter(CUDA::KernelInvocation &invocation, const CUDA::Vector<T>& values, const std::string& description) const
{
	if (Utils::Options::IsDebug_Print())
	{
		std::string valDescription;
		auto first = true;
		for (const auto& value : values)
		{
			if (!first)
			{
				valDescription += ", ";
			}
			first = false;
			valDescription += std::to_string(value);
		}
		Utils::Logger::LogDebug("Initializing list input argument: " + description + " [" + std::to_string(sizeof(T)) + " bytes x " + std::to_string(values.size()) + " = {" + valDescription + "}]");
	}

	auto buffer = CUDA::BufferManager::CreateConstantBuffer(values.size() * sizeof(T));
	buffer->SetCPUBuffer(values.data());
	buffer->AllocateOnGPU();
	buffer->TransferToGPU();
	invocation.AddParameter(*buffer);

	return buffer;
}

}
}
