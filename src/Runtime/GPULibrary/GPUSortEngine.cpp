#include "Runtime/GPULibrary/GPUSortEngine.h"

#include "Analysis/Shape/Shape.h"

#include "CUDA/Buffer.h"
#include "CUDA/Kernel.h"
#include "CUDA/KernelInvocation.h"
#include "CUDA/Module.h"

#include "HorseIR/Modules/GPUModule.h"
#include "HorseIR/Semantics/SemanticAnalysis.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "PTX/Program.h"

#include "Runtime/JITCompiler.h"

#include "Utils/Logger.h"
#include "Utils/Math.h"
#include "Utils/Options.h"

namespace Runtime {

VectorBuffer *GPUSortEngine::Sort(const std::vector<VectorBuffer *>& columns, const std::vector<char>& orders)
{
	// Generate the program (init and sort functions) for the provided columns

	auto [program, initFunction, sortFunction] = GenerateProgram(columns, orders);

	// Compute the size and active size (next highest power of 2) of the colums used for bitonic sort

	std::vector<const Analysis::VectorShape *> dataShapes;

	auto size = 0;
	bool first = true;
	for (const auto column : columns)
	{
		if (first)
		{
			size = column->GetElementCount();
			first = false;
		}
		else if (size != column->GetElementCount())
		{
			Utils::Logger::LogError("Sort requires all columns have equal size [" + std::to_string(size) + " != " + std::to_string(column->GetElementCount()) + "]");
		}
		dataShapes.push_back(column->GetShape());
	}

	auto activeSize = Utils::Math::Power2(size);
	auto vectorShape = new Analysis::VectorShape(new Analysis::Shape::ConstantSize(activeSize));

	// Compile the HorseIR to PTX code using the current device

	auto& gpu = m_runtime.GetGPUManager();
	auto& device = gpu.GetCurrentDevice();

	Codegen::TargetOptions targetOptions;
	targetOptions.ComputeCapability = device->GetComputeCapability();
	targetOptions.MaxBlockSize = device->GetMaxThreadsDimension(0);
	targetOptions.WarpSize = device->GetWarpSize();

	// Construct the input options for the init and sort functions

	auto [initOptions, sortOptions] = GenerateInputOptions(vectorShape, dataShapes, initFunction, sortFunction);

	// Compile!

	JITCompiler compiler(targetOptions);
	auto ptxProgram = compiler.Compile({initFunction, sortFunction}, {&initOptions, &sortOptions});

	// Create and load the CUDA module for the program

	auto cModule = gpu.AssembleProgram(ptxProgram);

	// Create the init kernel

	unsigned int initArgumentCount = initFunction->GetParameterCount() + initFunction->GetReturnCount();
	CUDA::Kernel initKernel(initFunction->GetName(), initArgumentCount, cModule);

	auto& initKernelOptions = ptxProgram->GetEntryFunction(initFunction->GetName())->GetOptions();

	Utils::Logger::LogInfo("Generated program for function '" + initFunction->GetName() + "' with options");
	Utils::Logger::LogInfo(initKernelOptions.ToString(), 1);

	// Create the sort kernel (2 extra parameters for stage and substage)

	unsigned int sortArgumentCount = sortFunction->GetParameterCount() + sortFunction->GetReturnCount() + 2;
	CUDA::Kernel sortKernel(sortFunction->GetName(), sortArgumentCount, cModule);

	auto& sortKernelOptions = ptxProgram->GetEntryFunction(sortFunction->GetName())->GetOptions();

	Utils::Logger::LogInfo("Generated program for function '" + sortFunction->GetName() + "' with options");
	Utils::Logger::LogInfo(sortKernelOptions.ToString(), 1);

	Utils::Logger::LogInfo("Continuing program execution");

	// Execute the compiled kernels on the GPU
	//   1. Initialize the arguments
	//   2. Create the invocation (thread sizes + arguments)
	//   3. Execute
	//   [4. Get return values]

	// Initialize sort buffers with the padded vector size

	auto indexBuffer = VectorBuffer::Create(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), vectorShape);
	Utils::Logger::LogInfo("Initializing sort buffer: [" + indexBuffer->Description() + "]");

	std::vector<VectorBuffer *> sortBuffers;
	for (auto column : columns)
	{
		// Create a new buffer for the padded sort value

		auto buffer = VectorBuffer::Create(column->GetType(), vectorShape);

		Utils::Logger::LogInfo("Initializing sort buffer: [" + buffer->Description() + "]");

		sortBuffers.push_back(buffer);
	}

	// Compute the block sizes for the init bitonic kernel

	const auto blockSize = (activeSize < targetOptions.MaxBlockSize) ? activeSize : targetOptions.MaxBlockSize;
	const auto blockCount = activeSize / blockSize;

	// Perform the sort initialization

	CUDA::KernelInvocation initInvocation(initKernel);
	initInvocation.SetBlockShape(blockSize, 1, 1);
	initInvocation.SetGridShape(blockCount, 1, 1);

	// Transfer the buffers to the GPU, data as read, sort as write for immutability

	for (auto dataBuffer : columns)
	{
		initInvocation.AddParameter(*dataBuffer->GetGPUReadBuffer());
	}

	initInvocation.AddParameter(*indexBuffer->GetGPUWriteBuffer());
	for (auto sortBuffer : sortBuffers)
	{
		initInvocation.AddParameter(*sortBuffer->GetGPUWriteBuffer());
	}

	initInvocation.SetDynamicSharedMemorySize(initKernelOptions.GetDynamicSharedMemorySize());
	initInvocation.Launch();

	// Perform the iterative sort

	const auto iterations = static_cast<unsigned int>(std::log2(activeSize));
	for (auto stage = 0u; stage < iterations; ++stage)
	{
		for (auto substage = 0u; substage <= stage; ++substage)
		{
			// Compute he block sizes for the sort bitonic kernel

			const auto swapThreads = activeSize / 2;
			const auto blockSize = (swapThreads < targetOptions.MaxBlockSize) ? swapThreads : targetOptions.MaxBlockSize;
			const auto blockCount = swapThreads / blockSize;

			CUDA::KernelInvocation sortInvocation(sortKernel);
			sortInvocation.SetBlockShape(blockSize, 1, 1);
			sortInvocation.SetGridShape(blockCount, 1, 1);

			// Transfer the buffers to the GPU

			sortInvocation.AddParameter(*indexBuffer->GetGPUWriteBuffer());
			for (auto sortBuffer : sortBuffers)
			{
				sortInvocation.AddParameter(*sortBuffer->GetGPUWriteBuffer());
			}

			// Add stage and substage custom parameters

			auto stageConstant = new CUDA::TypedConstant<std::uint32_t>(stage);
			sortInvocation.AddParameter(*stageConstant);

			auto substageConstant = new CUDA::TypedConstant<std::uint32_t>(substage);
			sortInvocation.AddParameter(*substageConstant);

			sortInvocation.SetDynamicSharedMemorySize(sortKernelOptions.GetDynamicSharedMemorySize());
			sortInvocation.Launch();
		}
	}
	// Free the sort buffers

	for (auto sortBuffer : sortBuffers)
	{
		delete sortBuffer->GetGPUWriteBuffer();
	}

	// Resize the index buffer to fit the number of indices if needed

	if (activeSize > size)
	{
		// Allocate a smaller buffer and copy the data

		auto collapsedShape = new Analysis::VectorShape(new Analysis::Shape::ConstantSize(size));
		auto collapsedBuffer = VectorBuffer::Create(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), collapsedShape);

		CUDA::Buffer::Copy(collapsedBuffer->GetGPUWriteBuffer(), indexBuffer->GetGPUReadBuffer(), size * sizeof(std::int64_t));
		delete indexBuffer;
	
		return collapsedBuffer;
	}
	return indexBuffer;
}

std::pair<Codegen::InputOptions, Codegen::InputOptions> GPUSortEngine::GenerateInputOptions(
	const Analysis::VectorShape *vectorShape, const std::vector<const Analysis::VectorShape *>& dataShapes,
	const HorseIR::Function *initFunction, const HorseIR::Function *sortFunction
) const
{
	// Construct the input options for the init and sort functions

	Codegen::InputOptions initOptions;
	initOptions.ThreadGeometry = vectorShape;
	initOptions.ReturnShapes.push_back(vectorShape);

	auto paramIndex = 0u;
	for (const auto& parameter : initFunction->GetParameters())
	{
		initOptions.ParameterShapes[parameter->GetSymbol()] = dataShapes.at(paramIndex++);
		initOptions.ReturnShapes.push_back(vectorShape);
	}

	Codegen::InputOptions sortOptions;
	sortOptions.ThreadGeometry = vectorShape;

	for (const auto& parameter : sortFunction->GetParameters())
	{
		sortOptions.ParameterShapes[parameter->GetSymbol()] = vectorShape;
	}

	return {initOptions, sortOptions};
}

std::tuple<HorseIR::Program *, HorseIR::Function *, HorseIR::Function *> GPUSortEngine::GenerateProgram(const std::vector<VectorBuffer *>& columns, const std::vector<char>& orders) const
{
	std::vector<const HorseIR::BasicType *> columnTypes;
	for (auto column : columns)
	{
		columnTypes.push_back(column->GetType());
	}

	// Initialize the index and padded buffers

	auto initFunction = GenerateInitFunction(columnTypes, orders);
	auto sortFunction = GenerateSortFunction(columnTypes, orders);
	auto importDirective = new HorseIR::ImportDirective("GPU", "*");

	auto program = new HorseIR::Program({HorseIR::GPUModule, new HorseIR::Module("order", {importDirective, initFunction, sortFunction})});

	if (Utils::Options::Present(Utils::Options::Opt_Print_library))
	{
		// Pretty print the library HorseIR program

		Utils::Logger::LogInfo("Library program: order");

		auto programString = HorseIR::PrettyPrinter::PrettyString(program);
		Utils::Logger::LogInfo(programString, 0, true, Utils::Logger::NoPrefix);
	}

	HorseIR::SemanticAnalysis::Analyze(program);

	return {program, initFunction, sortFunction};
}

HorseIR::Function *GPUSortEngine::GenerateInitFunction(const std::vector<const HorseIR::BasicType *>& types, const std::vector<char>& orders) const
{
	std::vector<HorseIR::Parameter *> parameters;

	std::vector<HorseIR::Operand *> operands;
	std::vector<HorseIR::LValue *> lvalues;

	std::vector<HorseIR::Operand *> returnOperands;
	std::vector<HorseIR::Type *> returnTypes;

	lvalues.push_back(new HorseIR::VariableDeclaration("index", new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64)));
	returnOperands.push_back(new HorseIR::Identifier("index"));
	returnTypes.push_back(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64));

	auto index = 0u;
	for (const auto type : types)
	{
		auto name = "data_" + std::to_string(index++);
		parameters.push_back(new HorseIR::Parameter(name, type->Clone()));

		operands.push_back(new HorseIR::Identifier(name));
		lvalues.push_back(new HorseIR::VariableDeclaration(name + "_out", type->Clone()));

		returnOperands.push_back(new HorseIR::Identifier(name + "_out"));
		returnTypes.push_back(type->Clone());
	}
	operands.push_back(new HorseIR::BooleanLiteral(orders));

	auto initCall = new HorseIR::CallExpression(new HorseIR::FunctionLiteral(new HorseIR::Identifier("GPU", "order_init")), operands);
	auto initStatement = new HorseIR::AssignStatement(lvalues, initCall);
	auto returnStatement = new HorseIR::ReturnStatement(returnOperands);
	
	return new HorseIR::Function("order_init", parameters, returnTypes, {initStatement, returnStatement}, true);
}

HorseIR::Function *GPUSortEngine::GenerateSortFunction(const std::vector<const HorseIR::BasicType *>& types, const std::vector<char>& orders) const
{
	std::vector<HorseIR::Parameter *> parameters;
	std::vector<HorseIR::Operand *> operands;

	parameters.push_back(new HorseIR::Parameter("index", new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64)));
	operands.push_back(new HorseIR::Identifier("index"));

	auto index = 0u;
	for (const auto type : types)
	{
		auto name = "data_" + std::to_string(index++);
		parameters.push_back(new HorseIR::Parameter(name, type->Clone()));
		operands.push_back(new HorseIR::Identifier(name));
	}
	operands.push_back(new HorseIR::BooleanLiteral(orders));

	auto sortCall = new HorseIR::CallExpression(new HorseIR::FunctionLiteral(new HorseIR::Identifier("GPU", "order")), operands);
	auto sortStatement = new HorseIR::ExpressionStatement(sortCall);
	
	return new HorseIR::Function("order", parameters, {}, {sortStatement}, true);
}

}
