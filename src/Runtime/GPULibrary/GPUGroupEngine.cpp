#include "Runtime/GPULibrary/GPUGroupEngine.h"

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
#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/DataBuffers/ListBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"
#include "Runtime/GPULibrary/GPUSortEngine.h"

#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Runtime {

DictionaryBuffer *GPUGroupEngine::Group(const std::vector<VectorBuffer *>& dataBuffers)
{
	// Run the sort program first before grouping

	std::vector<char> orders;
	for (auto i = 0u; i < dataBuffers.size(); ++i)
	{
		orders.push_back(1);
	}

	GPUSortEngine sortEngine(m_runtime);
	auto [indexBuffer, sortedDataBuffers] = sortEngine.Sort(dataBuffers, orders);

	// Generate the program and group function for the provided buffers

	auto [program, groupFunction] = GenerateProgram(sortedDataBuffers);

	// Generate the shape of the group data

	auto size = 0;
	bool first = true;
	for (const auto dataBuffer : sortedDataBuffers)
	{
		if (first)
		{
			size = dataBuffer->GetElementCount();
			first = false;
		}
		else if (size != dataBuffer->GetElementCount())
		{
			Utils::Logger::LogError("Group requires all vectors have equal size [" + std::to_string(size) + " != " + std::to_string(dataBuffer->GetElementCount()) + "]");
		}
	}

	auto vectorShape = new Analysis::VectorShape(new Analysis::Shape::ConstantSize(size));

	// Compile the HorseIR to PTX code using the current device

	auto& gpu = m_runtime.GetGPUManager();
	auto& device = gpu.GetCurrentDevice();

	Codegen::TargetOptions targetOptions;
	targetOptions.ComputeCapability = device->GetComputeCapability();
	targetOptions.MaxBlockSize = device->GetMaxThreadsDimension(0);
	targetOptions.WarpSize = device->GetWarpSize();

	// Construct the input options for the group function

	auto inputOptions = GenerateInputOptions(vectorShape, groupFunction);

	// Compile!

	JITCompiler compiler(targetOptions);
	auto ptxProgram = compiler.Compile({groupFunction}, {&inputOptions});

	// Create and load the CUDA module for the program

	auto cModule = gpu.AssembleProgram(ptxProgram);

	// Create the group kernel (2 dynamic size parameters)

	unsigned int argumentCount = groupFunction->GetParameterCount() + groupFunction->GetReturnCount() + 2;
	CUDA::Kernel groupKernel(groupFunction->GetName(), argumentCount, cModule);

	auto& kernelOptions = ptxProgram->GetEntryFunction(groupFunction->GetName())->GetOptions();

	Utils::Logger::LogInfo("Generated program for function '" + groupFunction->GetName() + "' with options");
	Utils::Logger::LogInfo(kernelOptions.ToString(), 1);

	Utils::Logger::LogInfo("Continuing program execution");

	// Execute the compiled kernels on the GPU
	//   1. Initialize the arguments
	//   2. Create the invocation (thread sizes + arguments)
	//   3. Execute
	//   [4. Get return values]

	// Compute the thread geometry for the kernel

	const auto blockSize = (size < targetOptions.MaxBlockSize) ? size : targetOptions.MaxBlockSize;
	const auto blockCount = (size + blockSize - 1) / blockSize;

	// Perform the group

	CUDA::KernelInvocation invocation(groupKernel);
	invocation.SetBlockShape(blockSize, 1, 1);
	invocation.SetGridShape(blockCount, 1, 1);

	// Transfer the buffers to the GPU, data as read, sort as write for immutability

	Utils::Logger::LogInfo("Initializing index buffer: [" + indexBuffer->Description() + "]");
	invocation.AddParameter(*indexBuffer->GetGPUReadBuffer());

	for (auto dataBuffer : sortedDataBuffers)
	{
		Utils::Logger::LogInfo("Initializing data buffer: [" + dataBuffer->Description() + "]");
		invocation.AddParameter(*dataBuffer->GetGPUReadBuffer());
	}

	auto keysBuffer = VectorBuffer::Create(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), vectorShape->GetSize());
	Utils::Logger::LogInfo("Initializing keys buffer: [" + keysBuffer->Description() + "]");
	invocation.AddParameter(*keysBuffer->GetGPUReadBuffer());

	Utils::Logger::LogInfo("Initializing keys size buffer: [i32(" + std::to_string(sizeof(std::uint32_t)) + " bytes) x 1]");

	auto keysSize = 0u;
	auto valuesSize = 0u;

	auto keysSizeBuffer = new CUDA::Buffer(&keysSize, sizeof(std::uint32_t));
	keysSizeBuffer->AllocateOnGPU();
	keysSizeBuffer->Clear();
	invocation.AddParameter(*keysSizeBuffer);

	auto valuesBuffer = VectorBuffer::Create(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), vectorShape->GetSize());
	Utils::Logger::LogInfo("Initializing values buffer: [" + valuesBuffer->Description() + "]");
	invocation.AddParameter(*valuesBuffer->GetGPUReadBuffer());

	Utils::Logger::LogInfo("Initializing values size buffer: [i32(" + std::to_string(sizeof(std::uint32_t)) + " bytes) x 1]");

	auto valuesSizeBuffer = new CUDA::Buffer(&valuesSize, sizeof(std::uint32_t));
	valuesSizeBuffer->AllocateOnGPU();
	valuesSizeBuffer->Clear();
	invocation.AddParameter(*valuesSizeBuffer);
	
	invocation.SetDynamicSharedMemorySize(kernelOptions.GetDynamicSharedMemorySize());
	invocation.Launch();

	keysSizeBuffer->TransferToCPU();
	valuesSizeBuffer->TransferToCPU();

	if (keysSize != valuesSize)
	{
		Utils::Logger::LogError("Keys and values size mismatch forming @group dictionary [" + std::to_string(keysSize) + " != " + std::to_string(valuesSize) + "]");
	}

	auto values = BufferUtils::GetVectorBuffer<std::int64_t>(valuesBuffer)->GetCPUReadBuffer();
	auto indexes = BufferUtils::GetVectorBuffer<std::int64_t>(indexBuffer)->GetCPUReadBuffer()->GetValues();

	Utils::Logger::LogInfo("Initializing dictionary buffer: [entries = " + std::to_string(keysSize) + "]");

	std::vector<DataBuffer *> entryBuffers;
	for (auto entryIndex = 0; entryIndex < keysSize; ++entryIndex)
	{
		auto offset = values->GetValue(entryIndex);
		auto end = ((entryIndex + 1) == keysSize) ? size : values->GetValue(entryIndex + 1);
		auto entrySize = end - offset;

		CUDA::Vector<std::int64_t> data;
		data.insert(std::begin(data), std::begin(indexes) + offset, std::begin(indexes) + offset + entrySize);

		auto entryType = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64);
		auto entryData = new TypedVectorData<std::int64_t>(entryType, data);
		auto entryBuffer = new TypedVectorBuffer<std::int64_t>(entryData);

		Utils::Logger::LogInfo("Initializing entry " + std::to_string(entryIndex) + " buffer: [" + entryBuffer->Description() + "]");

		entryBuffers.push_back(entryBuffer);
	}

	// Collapse the index buffer

	auto collapsedKeysBuffer = VectorBuffer::Create(indexBuffer->GetType(), new Analysis::Shape::ConstantSize(keysSize));
	CUDA::Buffer::Copy(collapsedKeysBuffer->GetGPUWriteBuffer(), indexBuffer->GetGPUReadBuffer(), keysSize * indexBuffer->GetElementSize());

	auto dictionaryValuesBuffer = new ListBuffer(entryBuffers);
	auto dictionaryBuffer = new DictionaryBuffer(collapsedKeysBuffer, dictionaryValuesBuffer);

	// Delete all old buffers used to sort

	delete indexBuffer;
	for (auto buffer : sortedDataBuffers)
	{
		delete buffer;
	}

	return {dictionaryBuffer};
}

Codegen::InputOptions GPUGroupEngine::GenerateInputOptions(const Analysis::VectorShape *vectorShape, const HorseIR::Function *groupFunction) const
{
	// Construct the input options for the group function

	Codegen::InputOptions groupOptions;
	groupOptions.ThreadGeometry = vectorShape;

	auto predicate = new HorseIR::Identifier("index");
	groupOptions.ReturnShapes.push_back(new Analysis::VectorShape(new Analysis::Shape::CompressedSize(predicate, vectorShape->GetSize())));
	groupOptions.ReturnShapes.push_back(new Analysis::VectorShape(new Analysis::Shape::CompressedSize(predicate, vectorShape->GetSize())));

	auto paramIndex = 0u;
	for (const auto& parameter : groupFunction->GetParameters())
	{
		groupOptions.ParameterShapes[parameter->GetSymbol()] = vectorShape;
	}

	return groupOptions;
}

std::pair<HorseIR::Program *, HorseIR::Function *> GPUGroupEngine::GenerateProgram(const std::vector<VectorBuffer *>& dataBuffers) const
{
	std::vector<const HorseIR::BasicType *> dataTypes;
	for (auto dataBuffer : dataBuffers)
	{
		dataTypes.push_back(dataBuffer->GetType());
	}

	// Generate the group function and surrounding program

	auto groupFunction = GenerateGroupFunction(dataTypes);
	auto importDirective = new HorseIR::ImportDirective("GPU", "*");

	auto program = new HorseIR::Program({HorseIR::GPUModule, new HorseIR::Module("group", {importDirective, groupFunction})});

	if (Utils::Options::Present(Utils::Options::Opt_Print_library))
	{
		// Pretty print the library HorseIR program

		Utils::Logger::LogInfo("Library program: group");

		auto programString = HorseIR::PrettyPrinter::PrettyString(program);
		Utils::Logger::LogInfo(programString, 0, true, Utils::Logger::NoPrefix);
	}

	HorseIR::SemanticAnalysis::Analyze(program);

	return {program, groupFunction};
}

HorseIR::Function *GPUGroupEngine::GenerateGroupFunction(const std::vector<const HorseIR::BasicType *>& types) const
{
	std::vector<HorseIR::Parameter *> parameters;

	std::vector<HorseIR::Operand *> operands;
	std::vector<HorseIR::LValue *> lvalues;

	std::vector<HorseIR::Operand *> returnOperands;
	std::vector<HorseIR::Type *> returnTypes;

	auto keysType = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64);
	lvalues.push_back(new HorseIR::VariableDeclaration("keys", keysType));
	returnOperands.push_back(new HorseIR::Identifier("keys"));
	returnTypes.push_back(keysType->Clone());

	parameters.push_back(new HorseIR::Parameter("index", new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64)));
	operands.push_back(new HorseIR::Identifier("index"));

	auto index = 0u;
	for (const auto type : types)
	{
		auto name = "data_" + std::to_string(index++);
		parameters.push_back(new HorseIR::Parameter(name, type->Clone()));

		operands.push_back(new HorseIR::Identifier(name));
	}

	auto valuesType = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64);
	lvalues.push_back(new HorseIR::VariableDeclaration("values", valuesType));
	returnOperands.push_back(new HorseIR::Identifier("values"));
	returnTypes.push_back(valuesType->Clone());

	auto groupCall = new HorseIR::CallExpression(new HorseIR::FunctionLiteral(new HorseIR::Identifier("GPU", "group")), operands);
	auto groupStatement = new HorseIR::AssignStatement(lvalues, groupCall);
	auto returnStatement = new HorseIR::ReturnStatement(returnOperands);
	
	return new HorseIR::Function("group", parameters, returnTypes, {groupStatement, returnStatement}, true);
}

}
