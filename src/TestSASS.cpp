#include "Assembler/Assembler.h"
#include "Assembler/ELFGenerator.h"

#include "Backend/Scheduler/ListBlockScheduler.h"
#include "Backend/Scheduler/Profiles/Compute61Profile.h"
#include "Backend/Scheduler/Profiles/Compute86Profile.h"

#include "CUDA/BufferManager.h"
#include "CUDA/Compiler.h"
#include "CUDA/Device.h"
#include "CUDA/Module.h"
#include "CUDA/Kernel.h"
#include "CUDA/KernelInvocation.h"
#include "CUDA/Utils.h"

#include "Runtime/Runtime.h"

#include "SASS/Tree/Tree.h"
#include "SASS/Utils/PrettyPrinter.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Math.h"
#include "Utils/Options.h"

template<typename T>
void ExecuteTest(SASS::Program *sassProgram, const std::unique_ptr<CUDA::Device>& device)
{
	// Assembly program to binary

	Assembler::Assembler assembler;
	auto binaryProgram = assembler.Assemble(sassProgram);

	// Print debug

	auto programString = SASS::PrettyPrinter::PrettyString(sassProgram);
	Utils::Logger::LogInfo("Generated SASS program");
	Utils::Logger::LogInfo(programString, 0, true, Utils::Logger::NoPrefix);

	auto scheduledString = SASS::PrettyPrinter::PrettyString(sassProgram, true);
	Utils::Logger::LogInfo("Scheduled SASS program");
	Utils::Logger::LogInfo(scheduledString, 0, true, Utils::Logger::NoPrefix);

	auto assembledString = binaryProgram->ToString();
	Utils::Logger::LogInfo("Assembled SASS program");
	Utils::Logger::LogInfo(assembledString, 0, true, Utils::Logger::NoPrefix);

	// Generate ELF binary

	Assembler::ELFGenerator elfGenerator;
	auto elfProgram = elfGenerator.Generate(binaryProgram);

	CUDA::Compiler compiler;
	compiler.AddELFModule(*elfProgram);
	compiler.Compile(device);

	// Create kernel invocation and buffers

	CUDA::Module module(compiler.GetBinary(), compiler.GetBinarySize());
	CUDA::Kernel kernel("main", module);
	CUDA::KernelInvocation invocation(kernel);

	auto threadCount = 1;

	invocation.SetBlockShape(threadCount, 1, 1);
	invocation.SetGridShape(1, 1, 1);

	auto dataBuffer = CUDA::BufferManager::CreateBuffer(threadCount * sizeof(T));

	T inputData[threadCount];
	inputData[0] = 1;

	dataBuffer->SetCPUBuffer(inputData);
	dataBuffer->AllocateOnGPU();
	dataBuffer->TransferToGPU();

	invocation.AddParameter(*dataBuffer);

	// Execute kernel

	invocation.Launch();
	CUDA::Synchronize();

	// Transfer result

	T outputData[threadCount];
	dataBuffer->SetCPUBuffer(outputData);
	dataBuffer->TransferToCPU();

	// Log result

	for (auto i = 0; i < threadCount; ++i)
	{
		Utils::Logger::LogInfo("[" + std::to_string(i) + "]: output=" + std::to_string(outputData[i]));
	}
}

SASS::Program *TestProgramMaxwell(unsigned int compute)
{
	// Generate SASS program

	auto program = new SASS::Program();
	program->SetComputeCapability(compute);

	auto function = new SASS::Function("main");

	function->AddParameter(8); // 0x140
	function->SetRegisters(5);

	auto block = new SASS::BasicBlock("BB0");
	function->AddBasicBlock(block);

	auto R2 = new SASS::Register(2);
	auto R3 = new SASS::Register(3);
	auto R4 = new SASS::Register(4);
	auto R6 = new SASS::Register(6);

	auto R2_3 = new SASS::Register(2, 2);

	// Compute thread index

	block->AddInstruction(new SASS::Maxwell::S2RInstruction(
		R4, new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_CTAID_X)
	));
	block->AddInstruction(new SASS::Maxwell::S2RInstruction(
		R2, new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_TID_X)
	));

	block->AddInstruction(new SASS::Maxwell::XMADInstruction(
		R3, R4, new SASS::Constant(0x0, 0x8), SASS::RZ,
		SASS::Maxwell::XMADInstruction::Mode::MRG, SASS::Maxwell::XMADInstruction::Flags::H1_B
	));
	block->AddInstruction(new SASS::Maxwell::XMADInstruction(R2, R4, new SASS::Constant(0x0, 0x8), R2));
	block->AddInstruction(new SASS::Maxwell::XMADInstruction(
		R4, R4, R3, R2, SASS::Maxwell::XMADInstruction::Mode::PSL,
		SASS::Maxwell::XMADInstruction::Flags::CBCC | SASS::Maxwell::XMADInstruction::Flags::H1_A |
		SASS::Maxwell::XMADInstruction::Flags::H1_B
	));

	// Compute address

	block->AddInstruction(new SASS::Maxwell::IADDInstruction(
		R2, R4, new SASS::Constant(0x0, 0x140), SASS::Maxwell::IADDInstruction::Flags::CC
	));
	block->AddInstruction(new SASS::Maxwell::IADDInstruction(
		R3, SASS::RZ, new SASS::Constant(0x0, 0x144), SASS::Maxwell::IADDInstruction::Flags::X
	));

	// Load, increment, store

	block->AddInstruction(new SASS::Maxwell::LDGInstruction(
		R4, new SASS::Address(R2_3), SASS::Maxwell::LDGInstruction::Type::U8,
		SASS::Maxwell::LDGInstruction::Cache::None, SASS::Maxwell::LDGInstruction::Flags::E
	));
	block->AddInstruction(new SASS::Maxwell::IADD32IInstruction(R6, R4, new SASS::I32Immediate(0x1)));
	block->AddInstruction(new SASS::Maxwell::STGInstruction(
		new SASS::Address(R2_3), R6, SASS::Maxwell::STGInstruction::Type::U8,
		SASS::Maxwell::STGInstruction::Cache::None, SASS::Maxwell::STGInstruction::Flags::E
	));
	block->AddInstruction(new SASS::Maxwell::EXITInstruction());

	// Schedule instructions

	Backend::Scheduler::Compute61Profile profile;
	Backend::Scheduler::ListBlockScheduler scheduler(profile);

	scheduler.Schedule(function);

	// Build program

	program->AddFunction(function);

	return program;
}

SASS::Program *TestProgramVolta(unsigned int compute)
{
	// Generate SASS program

	auto program = new SASS::Program();
	program->SetComputeCapability(compute);

	auto function = new SASS::Function("main");

	function->AddParameter(8); // 0x160
	function->SetRegisters(5 + 2); // 2 for per-thread PC

	auto block = new SASS::BasicBlock("BB0");
	function->AddBasicBlock(block);

	auto R2 = new SASS::Register(2);
	auto R3 = new SASS::Register(3);
	auto R4 = new SASS::Register(4);
	auto R6 = new SASS::Register(6);

	auto R2_3 = new SASS::Register(2, 2);

	auto P0 = new SASS::Predicate(0);

	// Compute thread index

	block->AddInstruction(new SASS::Volta::S2RInstruction(
		R4, new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_CTAID_X)
	));
	block->AddInstruction(new SASS::Volta::S2RInstruction(
		R2, new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_TID_X)
	));
	block->AddInstruction(new SASS::Volta::IMADInstruction(
		R4, R4, new SASS::Constant(0x0, 0x0), R2
	));

	// Compute address

	// IADD3 R2 , P0 , R4, c[0x0][0 x160], RZ ;
	// IADD3.X R3, RZ, c[0x0][0 x164], RZ, P0 , !PT ;

	block->AddInstruction(new SASS::Volta::IADD3Instruction(
		R2, P0, R4, new SASS::Constant(0x0, 0x160), SASS::RZ
	));
	block->AddInstruction(new SASS::Volta::IADD3Instruction(
		R3, SASS::RZ, new SASS::Constant(0x0, 0x164), SASS::RZ, P0, SASS::PT,
		SASS::Volta::IADD3Instruction::Flags::X | SASS::Volta::IADD3Instruction::Flags::NOT_E
	));

	// Load, increment, store

	block->AddInstruction(new SASS::Volta::LDGInstruction(
		R4, new SASS::Address(R2_3), SASS::Volta::LDGInstruction::Type::U8,
		SASS::Volta::LDGInstruction::Cache::None, SASS::Volta::LDGInstruction::Evict::None,
		SASS::Volta::LDGInstruction::Prefetch::None, SASS::Volta::LDGInstruction::Flags::E
	));
	block->AddInstruction(new SASS::Volta::IADD3Instruction(R6, R4, new SASS::I32Immediate(0x1), SASS::RZ));
	block->AddInstruction(new SASS::Volta::STGInstruction(
		new SASS::Address(R2_3), R6, SASS::Volta::STGInstruction::Type::U8,
		SASS::Volta::STGInstruction::Cache::None, SASS::Volta::STGInstruction::Evict::None,
		SASS::Volta::STGInstruction::Flags::E
	));
	block->AddInstruction(new SASS::Volta::EXITInstruction());

	// Schedule instructions

	Backend::Scheduler::Compute86Profile profile;
	Backend::Scheduler::ListBlockScheduler scheduler(profile);

	scheduler.Schedule(function);

	// Build program

	program->AddFunction(function);

	return program;
}

int main(int argc, const char *argv[])
{
	// Initialize the input arguments from the command line

	Utils::Options::Initialize(argc, argv);

	// Initialize the runtime environment and check that the machine is capable of running the query

	Utils::Chrono::Initialize();

	auto& runtime = *Runtime::Runtime::GetInstance();
	runtime.Initialize();

	auto& gpu = runtime.GetGPUManager();
	auto& device = gpu.GetCurrentDevice();

	// Execute tests

	auto compute = device->GetComputeMajor() * 10 + device->GetComputeMinor();
	if (SASS::Maxwell::IsSupported(compute))
	{
		ExecuteTest<std::uint8_t>(TestProgramMaxwell(compute), device);
	}
	else if (SASS::Volta::IsSupported(compute))
	{
		ExecuteTest<std::uint8_t>(TestProgramVolta(compute), device);
	}
	else
	{
		Utils::Logger::LogError("Unsupported CUDA compute capability " + device->GetComputeCapability());
	}

	// Cleanup

	Utils::Chrono::Complete();
}
