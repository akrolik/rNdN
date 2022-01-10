#include "Assembler/Assembler.h"
#include "Assembler/ELFGenerator.h"

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

template<class T = std::uint32_t>
std::uint32_t ExecuteTest(const std::string& name, const std::unique_ptr<CUDA::Device>& device, const std::pair<SASS::Program *, std::uint32_t>& test)
{
	// Generate ELF binary

	auto sassProgram = std::get<0>(test);
	auto sassCycles = std::get<1>(test);

	Assembler::Assembler assembler;
	auto binaryProgram = assembler.Assemble(sassProgram);

	Assembler::ELFGenerator elfGenerator;
	auto elfProgram = elfGenerator.Generate(binaryProgram);

	CUDA::Compiler compiler;
	compiler.AddELFModule(*elfProgram);
	compiler.Compile(device);

	// Create kernel invocation and buffers

	auto threadCount = SASS::WARP_SIZE;

	CUDA::Module module(compiler.GetBinary(), compiler.GetBinarySize());
	CUDA::Kernel kernel("main", module);
	CUDA::KernelInvocation invocation(kernel);

	invocation.SetBlockShape(threadCount, 1, 1);
	invocation.SetGridShape(1, 1, 1);
	invocation.SetDynamicSharedMemorySize(1024);

	auto cyclesBuffer = CUDA::BufferManager::CreateBuffer(threadCount * sizeof(std::uint32_t));
	auto valuesBuffer = CUDA::BufferManager::CreateBuffer(threadCount * sizeof(T));

	cyclesBuffer->AllocateOnGPU();
	cyclesBuffer->Clear();

	valuesBuffer->AllocateOnGPU();
	valuesBuffer->Clear();

	invocation.AddParameter(*cyclesBuffer);
	invocation.AddParameter(*valuesBuffer);

	// Execute kernel

	invocation.Launch();
	CUDA::Synchronize();

	// Transfer result

	std::uint32_t cycles[threadCount];
	cyclesBuffer->SetCPUBuffer(cycles);
	cyclesBuffer->TransferToCPU();

	T values[threadCount];
	valuesBuffer->SetCPUBuffer(values);
	valuesBuffer->TransferToCPU();

	// Log result

	Utils::Logger::LogInfo("Test: " + name + " (offset=" + std::to_string(sassCycles) + ")");

	for (auto i = 0; i < threadCount; ++i)
	{
		auto cycleCount = cycles[i] - sassCycles;
		Utils::Logger::LogInfo(
			"[" + std::to_string(i) + "]: cycles=" + std::to_string(cycleCount) + " (" + std::to_string(cycles[i]) + ")" +
			", value=" + std::to_string(values[i])
		);
	}
	return cycles[0] - sassCycles;
}

template<class T, class F>
std::pair<SASS::Program *, std::uint32_t> TestProgramMaxwell(const std::unique_ptr<CUDA::Device>& device, F benchmark, bool write = true)
{
	// Generate SASS program

	auto program = new SASS::Program();
	program->SetComputeCapability(device->GetComputeMajor() * 10 + device->GetComputeMinor());

	auto function = new SASS::Function("main");

	function->AddParameter(8); // 0x140
	function->AddParameter(8); // 0x148
	function->SetRegisters(32);

	auto block = new SASS::BasicBlock("BB0");
	function->AddBasicBlock(block);

	// Get registers

	auto R0 = new SASS::Register(0);
	auto R1 = new SASS::Register(1);
	auto R2 = new SASS::Register(2);
	auto R3 = new SASS::Register(3);
	auto R4 = new SASS::Register(4);
	auto R5 = new SASS::Register(5);
	auto R6 = new SASS::Register(6);
	auto R7 = new SASS::Register(7);
	auto R8 = new SASS::Register(8);
	auto R9 = new SASS::Register(10);

	auto R2_2 = new SASS::Register(2, 2);
	auto R4_2 = new SASS::Register(4, 2);

	auto R16 = new SASS::Register(16);
	auto R17 = new SASS::Register(17);
	auto R18 = new SASS::Register(18);
	auto R19 = new SASS::Register(19);

	auto P0 = new SASS::Predicate(0);
	auto P1 = new SASS::Predicate(1);

	// Get output address

	auto inst0 = new SASS::Maxwell::S2RInstruction(R0, new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_TID_X));
	auto inst1 = new SASS::Maxwell::SHLInstruction(R1, R0, new SASS::I32Immediate(Utils::Math::Log2(sizeof(std::uint32_t))));
	auto inst2 = new SASS::Maxwell::SHLInstruction(R9, R0, new SASS::I32Immediate(Utils::Math::Log2(sizeof(T))));

	auto inst3 = new SASS::Maxwell::IADDInstruction(R2, R1, new SASS::Constant(0x0, 0x140), SASS::Maxwell::IADDInstruction::Flags::CC);
	auto inst4 = new SASS::Maxwell::IADDInstruction(R3, SASS::RZ, new SASS::Constant(0x0, 0x144), SASS::Maxwell::IADDInstruction::Flags::X);

	auto inst5 = new SASS::Maxwell::IADDInstruction(R4, R9, new SASS::Constant(0x0, 0x148), SASS::Maxwell::IADDInstruction::Flags::CC);
	auto inst6 = new SASS::Maxwell::IADDInstruction(R5, SASS::RZ, new SASS::Constant(0x0, 0x14c), SASS::Maxwell::IADDInstruction::Flags::X);

	auto inst7 = new SASS::Maxwell::MOV32IInstruction(R18, new SASS::I32Immediate(0));
	auto inst8 = new SASS::Maxwell::MOV32IInstruction(R19, new SASS::I32Immediate(0));
	auto inst9 = new SASS::Maxwell::PSETPInstruction(
		P0, SASS::PT, SASS::PT, SASS::PT, SASS::PT,
		SASS::Maxwell::PSETPInstruction::BooleanOperator1::AND,
		SASS::Maxwell::PSETPInstruction::BooleanOperator2::AND
	);

	auto inst10 = new SASS::Maxwell::ISETPInstruction(
		P1, SASS::PT, R0, new SASS::I32Immediate(1), SASS::PT,
		SASS::Maxwell::ISETPInstruction::ComparisonOperator::LE,
		SASS::Maxwell::ISETPInstruction::BooleanOperator::AND,
		SASS::Maxwell::ISETPInstruction::Flags::U32
	);
	auto inst11 = new SASS::Maxwell::CS2RInstruction(R6, new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_CLOCKLO));

	inst0->GetSchedule().SetStall(2);
	inst1->GetSchedule().SetStall(6);
	inst2->GetSchedule().SetStall(6);
	inst3->GetSchedule().SetStall(6);
	inst4->GetSchedule().SetStall(6);
	inst5->GetSchedule().SetStall(6);
	inst6->GetSchedule().SetStall(6);
	inst7->GetSchedule().SetStall(6);
	inst8->GetSchedule().SetStall(6);
	inst9->GetSchedule().SetStall(13);
	inst9->GetSchedule().SetYield(false);
	inst10->GetSchedule().SetStall(13);
	inst10->GetSchedule().SetYield(false);
	inst11->GetSchedule().SetStall(6);

	inst0->GetSchedule().SetWriteBarrier(SASS::Schedule::Barrier::SB0);
	inst1->GetSchedule().SetWaitBarriers(SASS::Schedule::Barrier::SB0);

	block->AddInstruction(inst0);
	block->AddInstruction(inst1);
	block->AddInstruction(inst2);
	block->AddInstruction(inst3);
	block->AddInstruction(inst4);
	block->AddInstruction(inst5);
	block->AddInstruction(inst6);
	block->AddInstruction(inst7);
	block->AddInstruction(inst8);
	block->AddInstruction(inst9);
	block->AddInstruction(inst10);
	block->AddInstruction(inst11);

	// Instruction to benchmark

	auto extraCycles = benchmark(block, R16, R0, P0, P1);

	// Barrier wait

	auto barrier = new SASS::Maxwell::MOVInstruction(R18, R16);
	barrier->SetPredicate(P0);
	barrier->GetSchedule().SetStall(6);
	barrier->GetSchedule().SetWaitBarriers(SASS::Schedule::Barrier::SB0);
	block->AddInstruction(barrier);

	// Store the difference between clocks

	auto inst12 = new SASS::Maxwell::CS2RInstruction(R7, new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_CLOCKLO));
	auto inst13 = new SASS::Maxwell::IADDInstruction(R8, R7, R6, SASS::Maxwell::IADDInstruction::Flags::NEG_B);
	auto inst14 = new SASS::Maxwell::STGInstruction(
		new SASS::Address(R2_2), R8, SASS::Maxwell::STGInstruction::Type::X32,
		SASS::Maxwell::STGInstruction::Cache::None, SASS::Maxwell::STGInstruction::Flags::E
	);

	// Store value

	auto inst15 = new SASS::Maxwell::MOVInstruction(R19, R17);
	inst15->SetPredicate(P1);

	auto inst16 = new SASS::Maxwell::STGInstruction(
		new SASS::Address(R4_2), R18, (sizeof(T) <= 4 ? SASS::Maxwell::STGInstruction::Type::X32 : SASS::Maxwell::STGInstruction::Type::X64),
		SASS::Maxwell::STGInstruction::Cache::None, SASS::Maxwell::STGInstruction::Flags::E
	);
	auto inst17 = new SASS::Maxwell::EXITInstruction();

	inst12->GetSchedule().SetStall(6);
	inst13->GetSchedule().SetStall(6);
	inst14->GetSchedule().SetStall(2);
	inst15->GetSchedule().SetStall(6);
	inst16->GetSchedule().SetStall(2);
	inst17->GetSchedule().SetStall(15);

	block->AddInstruction(inst12);
	block->AddInstruction(inst13);
	block->AddInstruction(inst14);
	if (write)
	{
		block->AddInstruction(inst15);
		block->AddInstruction(inst16);
	}
	block->AddInstruction(inst17);

	program->AddFunction(function);

	return { program, 6 + 6 + extraCycles }; // CS2R + MOV
}

template<class T, class F>
std::pair<SASS::Program *, std::uint32_t> TestProgramVolta(const std::unique_ptr<CUDA::Device>& device, F benchmark, bool write = true)
{
	// Generate SASS program

	auto program = new SASS::Program();
	program->SetComputeCapability(device->GetComputeMajor() * 10 + device->GetComputeMinor());

	auto function = new SASS::Function("main");

	function->AddParameter(8); // 0x160
	function->AddParameter(8); // 0x168
	function->SetRegisters(32);

	auto block = new SASS::BasicBlock("BB0");
	function->AddBasicBlock(block);

	// Get registers

	auto R0 = new SASS::Register(0);
	auto R1 = new SASS::Register(1);
	auto R2 = new SASS::Register(2);
	auto R3 = new SASS::Register(3);
	auto R4 = new SASS::Register(4);
	auto R5 = new SASS::Register(5);
	auto R6 = new SASS::Register(6);
	auto R7 = new SASS::Register(7);
	auto R8 = new SASS::Register(8);
	auto R9 = new SASS::Register(9);

	auto R2_2 = new SASS::Register(2, 2);
	auto R4_2 = new SASS::Register(4, 2);

	auto R16 = new SASS::Register(16);
	auto R17 = new SASS::Register(17);
	auto R18 = new SASS::Register(18);
	auto R19 = new SASS::Register(19);

	auto P0 = new SASS::Predicate(0);
	auto P1 = new SASS::Predicate(1);
	auto P2 = new SASS::Predicate(2);

	// Get output address

	auto inst0 = new SASS::Volta::S2RInstruction(R0, new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_TID_X));
	auto inst1 = new SASS::Volta::SHFInstruction(
		R1, R0, new SASS::I32Immediate(Utils::Math::Log2(sizeof(std::uint32_t))), SASS::RZ,
		SASS::Volta::SHFInstruction::Direction::L, SASS::Volta::SHFInstruction::Type::U32
	);
	auto inst2 = new SASS::Volta::SHFInstruction(
		R9, R0, new SASS::I32Immediate(Utils::Math::Log2(sizeof(T))), SASS::RZ,
		SASS::Volta::SHFInstruction::Direction::L, SASS::Volta::SHFInstruction::Type::U32
	);

	auto inst3 = new SASS::Volta::IADD3Instruction(R2, P2, R1, new SASS::Constant(0x0, 0x160), SASS::RZ);
	auto inst4 = new SASS::Volta::IADD3Instruction(
		R3, SASS::RZ, new SASS::Constant(0x0, 0x164), SASS::RZ, P2, SASS::PT,
		SASS::Volta::IADD3Instruction::Flags::X | SASS::Volta::IADD3Instruction::Flags::NOT_E
	);

	auto inst5 = new SASS::Volta::IADD3Instruction(R4, P2, R9, new SASS::Constant(0x0, 0x168), SASS::RZ);
	auto inst6 = new SASS::Volta::IADD3Instruction(
		R5, SASS::RZ, new SASS::Constant(0x0, 0x16c), SASS::RZ, P2, SASS::PT,
		SASS::Volta::IADD3Instruction::Flags::X | SASS::Volta::IADD3Instruction::Flags::NOT_E
	);

	auto inst7 = new SASS::Volta::MOVInstruction(R18, new SASS::I32Immediate(0));
	auto inst8 = new SASS::Volta::MOVInstruction(R19, new SASS::I32Immediate(0));

	auto logicOperation = SASS::Volta::BinaryUtils::LogicOperation(
		[](std::uint8_t A, std::uint8_t B, std::uint8_t C)
		{
			return ((A & B) & C);
		}
	);
	auto inst9 = new SASS::Volta::PLOP3Instruction(
		P0, SASS::PT, SASS::PT, SASS::PT, SASS::PT,
		new SASS::I8Immediate(logicOperation), new SASS::I8Immediate(0x0)
	);

	auto inst10 = new SASS::Volta::ISETPInstruction(
		P1, SASS::PT, R0, new SASS::I32Immediate(1), SASS::PT,
		SASS::Volta::ISETPInstruction::ComparisonOperator::EQ,
		SASS::Volta::ISETPInstruction::BooleanOperator::AND,
		SASS::Volta::ISETPInstruction::Flags::U32
	);
	auto inst11 = new SASS::Volta::CS2RInstruction(R6, new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_CLOCKLO));

	inst0->GetSchedule().SetStall(2);
	inst1->GetSchedule().SetStall(4);
	inst2->GetSchedule().SetStall(4);
	inst3->GetSchedule().SetStall(4);
	inst4->GetSchedule().SetStall(4);
	inst5->GetSchedule().SetStall(4);
	inst6->GetSchedule().SetStall(4);
	inst7->GetSchedule().SetStall(4);
	inst8->GetSchedule().SetStall(4);
	inst9->GetSchedule().SetStall(15);
	inst9->GetSchedule().SetYield(false);
	inst10->GetSchedule().SetStall(15);
	inst10->GetSchedule().SetYield(false);
	inst11->GetSchedule().SetStall(4);

	inst0->GetSchedule().SetWriteBarrier(SASS::Schedule::Barrier::SB0);
	inst1->GetSchedule().SetWaitBarriers(SASS::Schedule::Barrier::SB0);

	block->AddInstruction(inst0);
	block->AddInstruction(inst1);
	block->AddInstruction(inst2);
	block->AddInstruction(inst3);
	block->AddInstruction(inst4);
	block->AddInstruction(inst5);
	block->AddInstruction(inst6);
	block->AddInstruction(inst7);
	block->AddInstruction(inst8);
	block->AddInstruction(inst9);
	block->AddInstruction(inst10);
	block->AddInstruction(inst11);

	// Instruction to benchmark

	auto extraCycles = benchmark(block, R16, R0, P0, P1);

	// Barrier wait

	auto barrier = new SASS::Volta::MOVInstruction(R18, R16);
	barrier->SetPredicate(P0);
	barrier->GetSchedule().SetStall(4);
	barrier->GetSchedule().SetWaitBarriers(SASS::Schedule::Barrier::SB0);
	block->AddInstruction(barrier);

	// Store the difference between clocks

	auto inst12 = new SASS::Volta::CS2RInstruction(R7, new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_CLOCKLO));
	auto inst13 = new SASS::Volta::IADD3Instruction(R8, R7, R6, SASS::RZ, SASS::Volta::IADD3Instruction::Flags::NEG_B);
	auto inst14 = new SASS::Volta::STGInstruction(
		new SASS::Address(R2_2), R8,
		SASS::Volta::STGInstruction::Type::X32,
		SASS::Volta::STGInstruction::Cache::None,
		SASS::Volta::STGInstruction::Evict::None,
		SASS::Volta::STGInstruction::Flags::E
	);

	// Store value

	auto inst15 = new SASS::Volta::MOVInstruction(R19, R17);
	inst15->SetPredicate(P0);

	auto inst16 = new SASS::Volta::STGInstruction(
		new SASS::Address(R4_2), R18,
		(sizeof(T) <= 4 ? SASS::Volta::STGInstruction::Type::X32 : SASS::Volta::STGInstruction::Type::X64),
		SASS::Volta::STGInstruction::Cache::None,
		SASS::Volta::STGInstruction::Evict::None,
		SASS::Volta::STGInstruction::Flags::E
	);
	auto inst17 = new SASS::Volta::EXITInstruction();

	inst12->GetSchedule().SetStall(4);
	inst13->GetSchedule().SetStall(4);
	inst14->GetSchedule().SetStall(2);
	inst15->GetSchedule().SetStall(4);
	inst16->GetSchedule().SetStall(2);
	inst17->GetSchedule().SetStall(15);

	block->AddInstruction(inst12);
	block->AddInstruction(inst13);
	block->AddInstruction(inst14);
	if (write)
	{
		block->AddInstruction(inst15);
		block->AddInstruction(inst16);
	}
	block->AddInstruction(inst17);

	program->AddFunction(function);

	return { program, 4 + 4 + extraCycles }; // CS2R + MOV
}

template<class T, class F>
void Test(const std::string& name, const std::unique_ptr<CUDA::Device>& device, F instruction, std::uint32_t stall, bool write = false, bool read = false)
{
	auto compute = device->GetComputeMajor() * 10 + device->GetComputeMinor();

	Utils::Logger::LogSection(name);
	Utils::Logger::LogSection("-----------------------------------", false);

	auto testDepth = [&](SASS::BasicBlock *block, SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS)
	{
		auto inst = instruction(RD, RS, PD, PS, false);
		inst->GetSchedule().SetStall(stall);
		inst->GetSchedule().SetYield(stall < 13);
		if (write)
		{
			inst->GetSchedule().SetWriteBarrier(SASS::Schedule::Barrier::SB0);
		}
		if (read)
		{
			inst->GetSchedule().SetReadBarrier(SASS::Schedule::Barrier::SB0);
		}
		block->AddInstruction(inst);

		return 0;
	};

	auto depthProgram = (SASS::Maxwell::IsSupported(compute) ? TestProgramMaxwell<T>(device, testDepth) : TestProgramVolta<T>(device, testDepth));
	auto offset = ExecuteTest<T>(name + " Depth", device, depthProgram);

	auto testThroughput = [&](SASS::BasicBlock *block, SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS)
	{
		auto inst0 = instruction(RD, RS, PD, PS, false);
		inst0->GetSchedule().SetStall(1);
		inst0->GetSchedule().SetYield(true);
		block->AddInstruction(inst0);

		auto inst1 = instruction(RD, RS, PD, PS, true);
		inst1->GetSchedule().SetStall(stall);
		inst1->GetSchedule().SetYield(stall < 13);
		if (write)
		{
			inst1->GetSchedule().SetWriteBarrier(SASS::Schedule::Barrier::SB0);
		}
		if (read)
		{
			inst1->GetSchedule().SetReadBarrier(SASS::Schedule::Barrier::SB0);
		}
		block->AddInstruction(inst1);

		return offset;
	};

	auto throughputProgram = (SASS::Maxwell::IsSupported(compute) ? TestProgramMaxwell<T>(device, testThroughput) : TestProgramVolta<T>(device, testThroughput));
	ExecuteTest<T>(name + " Throughput", device, throughputProgram);
}

void TestMaxwellSpecialRegister(const std::unique_ptr<CUDA::Device>& device)
{
	Test<std::uint32_t>("Special Register", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Maxwell::S2RInstruction(RD, new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_TID_X));
	}, 2, true);
}

void TestMaxwellInteger(const std::unique_ptr<CUDA::Device>& device)
{
	Test<std::uint32_t>("Integer", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Maxwell::IADDInstruction(RD, RS, new SASS::I32Immediate(1));
	}, 6);
}

void TestMaxwellSinglePrecision(const std::unique_ptr<CUDA::Device>& device)
{
	Test<float>("Single-Precision", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Maxwell::FADDInstruction(RD, RS, new SASS::F32Immediate(1.0));
	}, 6);
}

void TestMaxwellDoublePrecision(const std::unique_ptr<CUDA::Device>& device)
{
	Test<double>("Double-Precision (Write)", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Maxwell::DADDInstruction(RD, RS, new SASS::F64Immediate(1.0));
	}, 2, true);

	Test<double>("Double-Precision (Read)", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Maxwell::DADDInstruction(RD, RS, new SASS::F64Immediate(1.0));
	}, 2, false, true);
}

void TestMaxwellMUFU(const std::unique_ptr<CUDA::Device>& device)
{
	Test<float>("Special Function (Write)", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Maxwell::MUFUInstruction(RD, RS, SASS::Maxwell::MUFUInstruction::Function::COS);
	}, 2, true);

	Test<float>("Special Function (Read)", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Maxwell::MUFUInstruction(RD, RS, SASS::Maxwell::MUFUInstruction::Function::COS);
	}, 2, false, true);
}

void TestMaxwellConversion(const std::unique_ptr<CUDA::Device>& device)
{
	Test<std::uint32_t>("I2I", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Maxwell::I2IInstruction(
			RD, RS, SASS::Maxwell::I2IInstruction::DestinationType::S32, SASS::Maxwell::I2IInstruction::SourceType::S8
		);
	}, 2, true);

	Test<float>("I2F (32-bit)", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Maxwell::I2FInstruction(
			RD, RS, SASS::Maxwell::I2FInstruction::DestinationType::F32, SASS::Maxwell::I2FInstruction::SourceType::S32
		);
	}, 2, true);

	Test<std::uint32_t>("I2F (64-bit)", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Maxwell::I2FInstruction(
			RD, RS, SASS::Maxwell::I2FInstruction::DestinationType::F64, SASS::Maxwell::I2FInstruction::SourceType::S64
		);
	}, 2, true);

	// Test<std::uint32_t>("POPC", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	// {
	// 	return new SASS::Maxwell::POPCInstruction(RD, RS);
	// }, 2, true);

	// Test<std::uint32_t>("FLO", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	// {
	// 	return new SASS::Maxwell::FLOInstruction(RD, RS);
	// }, 2, true);
}
 
void TestMaxwellComparison(const std::unique_ptr<CUDA::Device>& device)
{
	Test<std::uint32_t>("PSETP", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Maxwell::PSETPInstruction(
			PD, SASS::PT, SASS::PT, PS, SASS::PT,
			SASS::Maxwell::PSETPInstruction::BooleanOperator1::AND,
			SASS::Maxwell::PSETPInstruction::BooleanOperator2::AND,
			SASS::Maxwell::PSETPInstruction::Flags::NOT_B
		);
	}, 13);

	Test<std::uint32_t>("ISETP", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Maxwell::ISETPInstruction(
			PD, SASS::PT, RS, new SASS::I32Immediate(1), SASS::PT,
			SASS::Maxwell::ISETPInstruction::ComparisonOperator::EQ,
			SASS::Maxwell::ISETPInstruction::BooleanOperator::AND,
			SASS::Maxwell::ISETPInstruction::Flags::U32
		);
	}, 13);

	Test<std::uint32_t>("DSETP", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Maxwell::DSETPInstruction(
			PD, SASS::PT, RS, RS, SASS::PT,
			SASS::Maxwell::DSETPInstruction::ComparisonOperator::EQ,
			SASS::Maxwell::DSETPInstruction::BooleanOperator::AND
		);
	}, 2, true);
}

void TestMaxwellShift(const std::unique_ptr<CUDA::Device>& device)
{
	Test<std::uint32_t>("Shift", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Maxwell::SHLInstruction(RD, RS, new SASS::I32Immediate(1));
	}, 6);

	// Test<std::uint32_t>("BFE", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	// {
	// 	return new SASS::Maxwell::BFEInstruction(RD, RS, new SASS::I32Immediate(1));
	// }, 6);
}
              
void TestMaxwellLoadStore(const std::unique_ptr<CUDA::Device>& device)
{
	Test<std::uint32_t>("Global Load (Write)", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		auto address = new SASS::Address(new SASS::Register(throughput ? 4 : 2));
		return new SASS::Maxwell::LDGInstruction(
			RD, address,
			SASS::Maxwell::LDGInstruction::Type::X32,
			SASS::Maxwell::LDGInstruction::Cache::None,
			SASS::Maxwell::LDGInstruction::Flags::E
		);
	}, 2, true);

	Test<std::uint32_t>("Global Load (Read)", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		auto address = new SASS::Address(new SASS::Register(throughput ? 4 : 2));
		return new SASS::Maxwell::LDGInstruction(
			RD, address,
			SASS::Maxwell::LDGInstruction::Type::X32,
			SASS::Maxwell::LDGInstruction::Cache::None,
			SASS::Maxwell::LDGInstruction::Flags::E
		);
	}, 2, false, true);

	Test<std::uint32_t>("Shared Load (Write)", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		auto address = new SASS::Address(SASS::RZ, throughput ? 512 : 0);
		return new SASS::Maxwell::LDSInstruction(
			RD, address, SASS::Maxwell::LDSInstruction::Type::X32
		);
	}, 2, true);

	Test<std::uint32_t>("Shared Load (Read)", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		auto address = new SASS::Address(SASS::RZ, throughput ? 512 : 0);
		return new SASS::Maxwell::LDSInstruction(
			RD, address, SASS::Maxwell::LDSInstruction::Type::X32
		);
	}, 2, false, true);

	Test<std::uint32_t>("Global Store", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		auto R4 = new SASS::Register(4);
		return new SASS::Maxwell::STGInstruction(
			new SASS::Address(R4), RS,
			SASS::Maxwell::STGInstruction::Type::X32,
			SASS::Maxwell::STGInstruction::Cache::None,
			SASS::Maxwell::STGInstruction::Flags::E
		);
	}, 2, false, true);

	ExecuteTest<std::uint32_t>("Global Store Delay", device, TestProgramMaxwell<std::uint32_t>(device,
		[&](SASS::BasicBlock *block, SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS)
		{
			auto R4 = new SASS::Register(4);
			auto R20 = new SASS::Register(20);

			auto inst0 = new SASS::Maxwell::MOVInstruction(R20, RS);
			inst0->GetSchedule().SetStall(2); // 4 delay
			block->AddInstruction(inst0);

			auto inst1 = new SASS::Maxwell::STGInstruction(
				new SASS::Address(R4), R20,
				SASS::Maxwell::STGInstruction::Type::X32,
				SASS::Maxwell::STGInstruction::Cache::None,
				SASS::Maxwell::STGInstruction::Flags::E
			);
			inst1->GetSchedule().SetStall(1);
			block->AddInstruction(inst1);

			return 1;
		}, false)
	);

	Test<std::uint32_t>("Shared Store", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		auto address = new SASS::Address(SASS::RZ, throughput ? 512 : 0);
		return new SASS::Maxwell::STSInstruction(
			address, RS, SASS::Maxwell::STSInstruction::Type::X32
		);
	}, 2, false, true);

	ExecuteTest<std::uint32_t>("Shared Store Delay", device, TestProgramMaxwell<std::uint32_t>(device,
		[&](SASS::BasicBlock *block, SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS)
		{
			auto R1 = new SASS::Register(1);
			auto R20 = new SASS::Register(20);

			auto inst0 = new SASS::Maxwell::MOVInstruction(R20, RS);
			inst0->GetSchedule().SetStall(4); // 2 delay
			block->AddInstruction(inst0);

			auto inst1 = new SASS::Maxwell::STSInstruction(
				new SASS::Address(R1), R20, SASS::Maxwell::STSInstruction::Type::X32
			);
			inst1->GetSchedule().SetStall(1);
			block->AddInstruction(inst1);

			auto inst2 = new SASS::Maxwell::LDSInstruction(
				RD, new SASS::Address(R1), SASS::Maxwell::LDSInstruction::Type::X32
			);
			inst2->GetSchedule().SetStall(2);
			inst2->GetSchedule().SetWriteBarrier(SASS::Schedule::Barrier::SB0);
			block->AddInstruction(inst2);

			return 0;
		})
	);

	Test<std::uint32_t>("Shuffle", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Maxwell::SHFLInstruction(
			SASS::PT, RD, RS, RS, RS, SASS::Maxwell::SHFLInstruction::ShuffleOperator::DOWN
		);
	}, 2, true);
}

void TestVoltaSpecialRegister(const std::unique_ptr<CUDA::Device>& device)
{
	Test<std::uint32_t>("Special Register", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Volta::S2RInstruction(RD, new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_TID_X));
	}, 2, true);
}

void TestVoltaInteger(const std::unique_ptr<CUDA::Device>& device)
{
	Test<std::uint32_t>("Integer", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Volta::IADD3Instruction(RD, RS, new SASS::I32Immediate(1), SASS::RZ);
	}, 5);

	Test<std::uint32_t>("Integer MAD", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Volta::IMADInstruction(RD, RS, new SASS::I32Immediate(1), SASS::RZ);
	}, 5);
}

void TestVoltaSinglePrecision(const std::unique_ptr<CUDA::Device>& device)
{
	Test<float>("Single-Precision", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Volta::FADDInstruction(RD, RS, new SASS::F32Immediate(1.0));
	}, 5);
}

void TestVoltaDoublePrecision(const std::unique_ptr<CUDA::Device>& device)
{
	Test<double>("Double-Precision (Write)", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Volta::DADDInstruction(RD, RS, new SASS::F64Immediate(1.0));
	}, 2, true);

	Test<double>("Double-Precision (Read)", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Volta::DADDInstruction(RD, RS, new SASS::F64Immediate(1.0));
	}, 2, false, true);
}

void TestVoltaMUFU(const std::unique_ptr<CUDA::Device>& device)
{
	Test<float>("Special Function (Write)", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Volta::MUFUInstruction(RD, RS, SASS::Volta::MUFUInstruction::Function::COS);
	}, 2, true);

	Test<float>("Special Function (Read)", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Volta::MUFUInstruction(RD, RS, SASS::Volta::MUFUInstruction::Function::COS);
	}, 2, false, true);
}

void TestVoltaConversion(const std::unique_ptr<CUDA::Device>& device)
{
	Test<std::uint16_t>("I2I", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Volta::I2IInstruction(RD, RS, SASS::Volta::I2IInstruction::DestinationType::U16);
	}, 2, true);

	Test<float>("I2F (32-bit)", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Volta::I2FInstruction(
			RD, RS, SASS::Volta::I2FInstruction::DestinationType::F32, SASS::Volta::I2FInstruction::SourceType::U32
		);
	}, 2, true);

	Test<double>("I2F (64-bit)", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Volta::I2FInstruction(
			RD, RS, SASS::Volta::I2FInstruction::DestinationType::F64, SASS::Volta::I2FInstruction::SourceType::S64
		);
	}, 2, true);
}

void TestVoltaComparison(const std::unique_ptr<CUDA::Device>& device)
{
	Test<std::uint32_t>("PLOP3", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		auto logicOperation = SASS::Volta::BinaryUtils::LogicOperation(
			[](std::uint8_t A, std::uint8_t B, std::uint8_t C)
			{
				return ((A & B) & C);
			}
		);
		return new SASS::Volta::PLOP3Instruction(
			PD, SASS::PT, PS, SASS::PT, SASS::PT,
			new SASS::I8Immediate(logicOperation), new SASS::I8Immediate(0x0)
		);
	}, 13);

	Test<std::uint32_t>("ISETP", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Volta::ISETPInstruction(
			PD, SASS::PT, RS, new SASS::I32Immediate(1), SASS::PT,
			SASS::Volta::ISETPInstruction::ComparisonOperator::LE,
			SASS::Volta::ISETPInstruction::BooleanOperator::AND,
			SASS::Volta::ISETPInstruction::Flags::U32
		);
	}, 13);

	Test<std::uint32_t>("DSETP", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Volta::DSETPInstruction(
			PD, SASS::PT, RS, SASS::RZ, SASS::PT,
			SASS::Volta::DSETPInstruction::ComparisonOperator::EQ,
			SASS::Volta::DSETPInstruction::BooleanOperator::AND
		);
	}, 2, true);
}

void TestVoltaShift(const std::unique_ptr<CUDA::Device>& device)
{
	Test<std::uint32_t>("Shift", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Volta::SHFInstruction(
			RD, RS, new SASS::I32Immediate(1), SASS::RZ,
			SASS::Volta::SHFInstruction::Direction::L,
			SASS::Volta::SHFInstruction::Type::U32
		);
	}, 5);
}
              
void TestVoltaLoadStore(const std::unique_ptr<CUDA::Device>& device)
{
	Test<std::uint32_t>("Global Load (Write)", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		auto address = new SASS::Address(new SASS::Register(throughput ? 4 : 2));
		return new SASS::Volta::LDGInstruction(
			RD, address,
			SASS::Volta::LDGInstruction::Type::X32,
			SASS::Volta::LDGInstruction::Cache::None,
			SASS::Volta::LDGInstruction::Evict::None,
			SASS::Volta::LDGInstruction::Prefetch::None,
			SASS::Volta::LDGInstruction::Flags::E
		);
	}, 2, true);

	Test<std::uint32_t>("Global Load (Read)", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		auto address = new SASS::Address(new SASS::Register(throughput ? 4 : 2));
		return new SASS::Volta::LDGInstruction(
			RD, address,
			SASS::Volta::LDGInstruction::Type::X32,
			SASS::Volta::LDGInstruction::Cache::None,
			SASS::Volta::LDGInstruction::Evict::None,
			SASS::Volta::LDGInstruction::Prefetch::None,
			SASS::Volta::LDGInstruction::Flags::E
		);
	}, 2, false, true);

	Test<std::uint32_t>("Shared Load (Write)", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		auto address = new SASS::Address(SASS::RZ, throughput ? 512 : 0);
		return new SASS::Volta::LDSInstruction(
			RD, address, SASS::Volta::LDSInstruction::Type::X32
		);
	}, 2, true);

	Test<std::uint32_t>("Shared Load (Read)", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		auto address = new SASS::Address(SASS::RZ, throughput ? 512 : 0);
		return new SASS::Volta::LDSInstruction(
			RD, address, SASS::Volta::LDSInstruction::Type::X32
		);
	}, 2, false, true);

	Test<std::uint32_t>("Global Store", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		auto R4 = new SASS::Register(4);
		return new SASS::Volta::STGInstruction(
			new SASS::Address(R4), RS,
			SASS::Volta::STGInstruction::Type::X32,
			SASS::Volta::STGInstruction::Cache::None,
			SASS::Volta::STGInstruction::Evict::None,
			SASS::Volta::STGInstruction::Flags::E
		);
	}, 2, false, true);

	ExecuteTest<std::uint32_t>("Global Store Delay", device, TestProgramVolta<std::uint32_t>(device,
		[&](SASS::BasicBlock *block, SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS)
		{
			auto R4 = new SASS::Register(4);
			auto R20 = new SASS::Register(20);

			auto inst0 = new SASS::Volta::MOVInstruction(R20, RS);
			inst0->GetSchedule().SetStall(4); // No delay
			block->AddInstruction(inst0);

			auto inst1 = new SASS::Volta::STGInstruction(
				new SASS::Address(R4), R20,
				SASS::Volta::STGInstruction::Type::X32,
				SASS::Volta::STGInstruction::Cache::None,
				SASS::Volta::STGInstruction::Evict::None,
				SASS::Volta::STGInstruction::Flags::E
			);
			inst1->GetSchedule().SetStall(1);
			block->AddInstruction(inst1);

			return 1;
		}, false)
	);

	Test<std::uint32_t>("Shared Store", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		auto address = new SASS::Address(SASS::RZ, throughput ? 512 : 0);
		return new SASS::Volta::STSInstruction(
			address, RS, SASS::Volta::STSInstruction::Type::X32
		);
	}, 2, false, true);

	ExecuteTest<std::uint32_t>("Shared Store Delay", device, TestProgramVolta<std::uint32_t>(device,
		[&](SASS::BasicBlock *block, SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS)
		{
			auto R1 = new SASS::Register(1);
			auto R20 = new SASS::Register(20);

			auto inst0 = new SASS::Volta::MOVInstruction(R20, RS);
			inst0->GetSchedule().SetStall(4); // No delay
			block->AddInstruction(inst0);

			auto inst1 = new SASS::Volta::STSInstruction(
				new SASS::Address(R1), R20, SASS::Volta::STSInstruction::Type::X32
			);
			inst1->GetSchedule().SetStall(1);
			block->AddInstruction(inst1);

			auto inst2 = new SASS::Volta::LDSInstruction(
				RD, new SASS::Address(R1), SASS::Volta::LDSInstruction::Type::X32
			);
			inst2->GetSchedule().SetStall(2);
			inst2->GetSchedule().SetWriteBarrier(SASS::Schedule::Barrier::SB0);
			block->AddInstruction(inst2);

			return 0;
		})
	);

	Test<std::uint32_t>("Shuffle", device, [](SASS::Register *RD, SASS::Register *RS, SASS::Predicate *PD, SASS::Predicate *PS, bool throughput)
	{
		return new SASS::Volta::SHFLInstruction(
			SASS::PT, RD, RS, RS, RS, SASS::Volta::SHFLInstruction::ShuffleOperator::DOWN
		);
	}, 2, true);
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
		TestMaxwellSpecialRegister(device);
		TestMaxwellInteger(device);
		TestMaxwellSinglePrecision(device);
		TestMaxwellDoublePrecision(device);
		TestMaxwellMUFU(device);
		TestMaxwellConversion(device);
		TestMaxwellComparison(device);
		TestMaxwellShift(device);
		TestMaxwellLoadStore(device);
	}
	else if (SASS::Volta::IsSupported(compute))
	{
		TestVoltaSpecialRegister(device);
		TestVoltaInteger(device);
		TestVoltaSinglePrecision(device);
		TestVoltaDoublePrecision(device);
		TestVoltaMUFU(device);
		TestVoltaConversion(device);
		TestVoltaComparison(device);
		TestVoltaShift(device);
		TestVoltaLoadStore(device);
	}
	else
	{
		Utils::Logger::LogError("Unsupported CUDA compute capability " + device->GetComputeCapability());
	}

	// Cleanup

	Utils::Chrono::Complete();
}
