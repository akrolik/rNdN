#include "Assembler/Assembler.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Assembler {

BinaryProgram *Assembler::Assemble(const SASS::Program *program)
{
	auto timeAssembler_start = Utils::Chrono::Start("SASS assembler");

	// Collect the device properties for codegen

	auto binaryProgram = new BinaryProgram();
	binaryProgram->SetComputeCapability(program->GetComputeCapability());
	binaryProgram->SetDynamicSharedMemory(program->GetDynamicSharedMemory());

	// Global variables

	for (const auto& globalVariable : program->GetGlobalVariables())
	{
		binaryProgram->AddGlobalVariable(globalVariable->GetName(), globalVariable->GetOffset(), globalVariable->GetSize());
	}

	// Assemble each function to binary

	for (const auto& function : program->GetFunctions())
	{
		binaryProgram->AddFunction(Assemble(function));
	}

	Utils::Chrono::End(timeAssembler_start);

	// Print assembled program with address and binary format

	if (Utils::Options::IsBackend_PrintAssembled())
	{
		Utils::Logger::LogInfo("Assembled SASS program");
		Utils::Logger::LogInfo(binaryProgram->ToString(), 0, true, Utils::Logger::NoPrefix);
	}

	return binaryProgram;
}

BinaryFunction *Assembler::Assemble(const SASS::Function *function)
{
	auto binaryFunction = new BinaryFunction(function->GetName());
	for (auto parameter : function->GetParameters())
	{
		binaryFunction->AddParameter(parameter);
	}

	std::unordered_map<std::string, unsigned int> blockIndex;
	auto blockOffset = 0u;

	// 1. Sequence basic blocks, create linear sequence of instruction
	// 2. Add self-loop BRA
	// 3. Pad instructions to multiple of 6
	// 4. Add 1 SCHI instruction for every 3 regular instructions (now padded to multiple of 8)
	// 5. Resolve branch targets
	//    (a) BRA instructions
	//    (b) SSY/PBK/PCNT instructions
	// 6. ELF properties
	//    (a) EXIT instructions
	//    (b) CTAID instructions
	//    (c) BAR instructions
	//    (d) SHFL instructions

	std::vector<const SASS::Instruction *> linearProgram;
	for (const auto block : function->GetBasicBlocks())
	{
		// Insert instructions from the basic block

		const auto& instructions = block->GetInstructions();
		linearProgram.insert(std::end(linearProgram), std::begin(instructions), std::end(instructions));

		// Keep track of the start address to resolve branches

		blockIndex.insert({block->GetName(), blockOffset});
		blockOffset += instructions.size();
	}

	// Insert self-loop

	auto selfName = "_END";
	auto selfBlock = new SASS::BasicBlock(selfName);
	auto selfBranch = new SASS::BRAInstruction(selfName);
	selfBranch->SetScheduling(15, true, 7, 7, 0, 0);

	linearProgram.push_back(selfBranch);
	blockIndex.insert({selfName, blockOffset++});

	// Padding to multiple of 6

	const auto PAD_SIZE = 6u;
	const auto pad = PAD_SIZE - (linearProgram.size() % PAD_SIZE);

	for (auto i = 0u; i < pad; i++)
	{
		// Insert NOPs with scheduling directive:
		//   0000 000000 111 111 0 0000

		auto nop = new SASS::NOPInstruction();
		nop->SetScheduling(0, false, 7, 7, 0, 0);
		linearProgram.push_back(nop);
	}

	// Construct SCHI instructions

	for (auto i = 0u; i < linearProgram.size(); i += 4)
	{
		auto inst1 = linearProgram.at(i);
		auto inst2 = linearProgram.at(i+1);
		auto inst3 = linearProgram.at(i+2);

		std::uint64_t assembled = 0u;
		assembled <<= 1;
		assembled |= inst3->GetScheduling().GenCode();
		assembled <<= 21;
		assembled |= inst2->GetScheduling().GenCode();
		assembled <<= 21;
		assembled |= inst1->GetScheduling().GenCode();

		linearProgram.insert(std::begin(linearProgram) + i, new SASS::SCHIInstruction(assembled));
	}

	// Resolve branches and control reconvergence instructions

	for (auto i = 0u; i < linearProgram.size(); ++i)
	{
		auto instruction = linearProgram.at(i);
		if (auto _branchInstruction = dynamic_cast<const SASS::BRAInstruction *>(instruction))
		{
			auto unpaddedIndex = blockIndex.at(_branchInstruction->GetTarget());
			auto paddedIndex = 1 + unpaddedIndex + (unpaddedIndex / 3);
			if (paddedIndex % 4 == 1)
			{
				paddedIndex--; // Begin at previous SCHI instruction
			}

			auto branchInstruction = const_cast<SASS::BRAInstruction *>(_branchInstruction);
			branchInstruction->SetTargetAddress(
				paddedIndex * sizeof(std::uint64_t),
				i * sizeof(std::uint64_t)
			);
		}
		else if (auto _divInstruction = dynamic_cast<const SASS::DivergenceInstruction *>(instruction))
		{
			auto unpaddedIndex = blockIndex.at(_divInstruction->GetTarget());
			auto paddedIndex = 1 + unpaddedIndex + (unpaddedIndex / 3);
			if (paddedIndex % 4 == 1)
			{
				paddedIndex--; // Begin at previous SCHI instruction
			}

			auto divInstruction = const_cast<SASS::DivergenceInstruction *>(_divInstruction);
			divInstruction->SetTargetAddress(
				paddedIndex * sizeof(std::uint64_t),
				i * sizeof(std::uint64_t)
			);
		}
	}
	
	// Collect special NVIDIA ELF properties

	constexpr auto MAX_BARRIERS = 16u;
	auto barriers = 0u;
	for (auto i = 0u; i < linearProgram.size(); ++i)
	{
		auto instruction = linearProgram.at(i);
		if (auto exitInstruction = dynamic_cast<const SASS::EXITInstruction *>(instruction))
		{
			binaryFunction->AddExitOffset(i * sizeof(std::uint64_t));
		}
		else if (auto s2rInstruction = dynamic_cast<const SASS::S2RInstruction *>(instruction))
		{
			auto kind = s2rInstruction->GetSource()->GetKind();
			if (kind == SASS::SpecialRegister::Kind::SR_CTAID_X ||
				kind == SASS::SpecialRegister::Kind::SR_CTAID_Y ||
				kind == SASS::SpecialRegister::Kind::SR_CTAID_X)
			{
				binaryFunction->AddS2RCTAIDOffset(i * sizeof(std::uint64_t));
			}
			if (kind == SASS::SpecialRegister::Kind::SR_CTAID_Z)
			{
				binaryFunction->SetCTAIDZUsed(true);
			}
		}
		else if (auto coopInstruction = dynamic_cast<const SASS::SHFLInstruction *>(instruction))
		{
			binaryFunction->AddCoopOffset(i * sizeof(std::uint64_t));
		}
		else if (auto barrierInstruction = dynamic_cast<const SASS::BARInstruction *>(instruction))
		{
			auto barrier = barrierInstruction->GetBarrier();
			if (dynamic_cast<const SASS::Register *>(barrier))
			{
				barriers = MAX_BARRIERS; // Max
			}
			else if (auto immediateBarrier = dynamic_cast<const SASS::I32Immediate *>(barrier))
			{
				auto value = immediateBarrier->GetValue();
				if (value + 1 > barriers)
				{
					barriers = value + 1;
				}
			}
		}
	}

	// Build relocations, resolving the address of each relocated instruction

	for (const auto& relocation : function->GetRelocations())
	{
		auto it = std::find(std::begin(linearProgram), std::end(linearProgram), relocation->GetInstruction());
		auto index = it - linearProgram.begin();
		auto address = index * sizeof(std::uint64_t);

		BinaryFunction::RelocationKind kind;
		switch (relocation->GetKind())
		{
			case SASS::Relocation::Kind::ABS32_LO_20:
			{
				kind = BinaryFunction::RelocationKind::ABS32_LO_20;
				break;
			}
			case SASS::Relocation::Kind::ABS32_HI_20:
			{
				kind = BinaryFunction::RelocationKind::ABS32_HI_20;
				break;
			}
			default:
			{
				Utils::Logger::LogError("Unknown relocation kind '" + SASS::Relocation::KindString(relocation->GetKind()) + "'");
			}
		}

		binaryFunction->AddRelocation(relocation->GetName(), address, kind);
	}

	// Setup barriers

	if (barriers > MAX_BARRIERS)
	{
		Utils::Logger::LogError("Barrier count " + std::to_string(barriers) + " exceeded maximum (" + std::to_string(MAX_BARRIERS) + ")");
	}
	binaryFunction->SetBarriers(barriers);

	// Assemble into binary

	auto binary = new std::uint64_t[linearProgram.size()];
	for (auto i = 0u; i < linearProgram.size(); ++i)
	{
		binary[i] = linearProgram.at(i)->ToBinary();
	}
	binaryFunction->SetText((char *)binary, sizeof(std::uint64_t) * linearProgram.size());
	binaryFunction->SetRegisters(function->GetRegisters());
	binaryFunction->SetLinearProgram(linearProgram);

	// Thread information

	if (auto [dimX, dimY, dimZ] = function->GetRequiredThreads(); dimX > 0)
	{
		binaryFunction->SetRequiredThreads(dimX, dimY, dimZ);
	}
	else if (auto [dimX, dimY, dimZ] = function->GetMaxThreads(); dimX > 0)
	{
		binaryFunction->SetMaxThreads(dimX, dimY, dimZ);
	}

	// Shared memory

	binaryFunction->SetSharedMemorySize(function->GetSharedMemorySize());

	// Constant memory

	binaryFunction->SetConstantMemory(function->GetConstantMemory());

	return binaryFunction;
}

}
