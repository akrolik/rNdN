#include "Assembler/Assembler.h"

#include <algorithm>
#include <cstring>
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

	// Global variables

	for (const auto& globalVariable : program->GetGlobalVariables())
	{
		binaryProgram->AddGlobalVariable(globalVariable->GetName(), globalVariable->GetSize(), globalVariable->GetDataSize());
	}
	
	// Shared variables

	for (const auto& sharedVariable : program->GetSharedVariables())
	{
		binaryProgram->AddSharedVariable(sharedVariable->GetName(), sharedVariable->GetSize(), sharedVariable->GetDataSize());
	}

	for (const auto& sharedVariable : program->GetDynamicSharedVariables())
	{
		binaryProgram->AddDynamicSharedVariable(sharedVariable->GetName());
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

	auto blockOffset = 0u;
	m_blockIndex.clear();

	std::vector<const SASS::Instruction *> linearProgram;
	for (const auto block : function->GetBasicBlocks())
	{
		// Insert instructions from the basic block

		const auto& instructions = block->GetInstructions();
		linearProgram.insert(std::end(linearProgram), std::begin(instructions), std::end(instructions));

		// Keep track of the start address to resolve branches

		m_blockIndex.insert({block->GetName(), blockOffset});
		blockOffset += instructions.size();
	}

	// Insert self-loop

	auto selfName = "_END";
	auto selfBlock = new SASS::BasicBlock(selfName);
	auto selfBranch = new SASS::BRAInstruction(selfName);
	selfBranch->SetSchedule(15, true, 7, 7, 0, 0);

	linearProgram.push_back(selfBranch);
	m_blockIndex.insert({selfName, blockOffset++});

	// Padding to multiple of 6

	const auto PAD_SIZE = 6u;
	const auto pad = PAD_SIZE - (linearProgram.size() % PAD_SIZE);

	for (auto i = 0u; i < pad; i++)
	{
		// Insert NOPs with scheduling directive:
		//   0000 000000 111 111 0 0000

		auto nop = new SASS::NOPInstruction();
		nop->SetSchedule(0, false, 7, 7, 0, 0);
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
		assembled |= inst3->GetSchedule().GenCode();
		assembled <<= 21;
		assembled |= inst2->GetSchedule().GenCode();
		assembled <<= 21;
		assembled |= inst1->GetSchedule().GenCode();

		linearProgram.insert(std::begin(linearProgram) + i, new SASS::SCHIInstruction(assembled));
	}

	// Resolve branches and control reconvergence instructions
	// Collect special NVIDIA ELF properties

	m_index = 0;
	m_barrierCount = 0;
	m_binaryFunction = binaryFunction;

	for (auto i = 0u; i < linearProgram.size(); ++i)
	{
		auto instruction = const_cast<SASS::Instruction *>(linearProgram.at(i));
		instruction->Accept(*this);

		m_index++;
	}

	// Setup barriers

	if (m_barrierCount > SASS::MAX_BARRIERS)
	{
		Utils::Logger::LogError("Barrier count " + std::to_string(m_barrierCount) + " exceeded maximum (" + std::to_string(SASS::MAX_BARRIERS) + ")");
	}
	binaryFunction->SetBarriers(m_barrierCount);

	// Build relocations, resolving the address of each relocated instruction

	for (const auto& relocation : function->GetRelocations())
	{
		auto it = std::find(std::begin(linearProgram), std::end(linearProgram), relocation->GetInstruction());
		auto address = (it - linearProgram.begin()) * sizeof(std::uint64_t);

		BinaryFunction::RelocationKind kind;
		switch (relocation->GetKind())
		{
			case SASS::Relocation::Kind::ABS24_20:
			{
				kind = BinaryFunction::RelocationKind::ABS24_20;
				break;
			}
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

	// Build indirect branches, resolving the target of each SYNC instruction

	for (const auto& indirectBranch : function->GetIndirectBranches())
	{
		// Compute offset of branch (SYNC) instruction

		auto it = std::find(std::begin(linearProgram), std::end(linearProgram), indirectBranch->GetBranch());
		auto offset = (it - linearProgram.begin())  * sizeof(std::uint64_t);

		// Compute offset of target block

		auto unpaddedIndex = m_blockIndex.at(indirectBranch->GetTarget());
		auto paddedIndex = 1 + unpaddedIndex + (unpaddedIndex / 3);
		if (paddedIndex % 4 == 1)
		{
			paddedIndex--; // Begin at previous SCHI instruction
		}
		auto target = paddedIndex * sizeof(std::uint64_t);

		// Add indirect branch

		binaryFunction->AddIndirectBranch(offset, target);
	}

	// Stack size

	binaryFunction->SetCRSStackSize(function->GetCRSStackSize());

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

	for (auto sharedVariable : function->GetSharedVariables())
	{
		binaryFunction->AddSharedVariable(sharedVariable->GetName(), sharedVariable->GetSize(), sharedVariable->GetDataSize());
	}

	// Constant memory

	binaryFunction->SetConstantMemory(function->GetConstantMemory());

	return binaryFunction;
}

// Address resolution

void Assembler::Visit(SASS::DivergenceInstruction *instruction)
{
	auto unpaddedIndex = m_blockIndex.at(instruction->GetTarget());
	auto paddedIndex = 1 + unpaddedIndex + (unpaddedIndex / 3);
	if (paddedIndex % 4 == 1)
	{
		paddedIndex--; // Begin at previous SCHI instruction
	}

	instruction->SetTargetAddress(
		paddedIndex * sizeof(std::uint64_t),
		m_index * sizeof(std::uint64_t)
	);
}

void Assembler::Visit(SASS::BRAInstruction *instruction)
{
	auto unpaddedIndex = m_blockIndex.at(instruction->GetTarget());
	auto paddedIndex = 1 + unpaddedIndex + (unpaddedIndex / 3);
	if (paddedIndex % 4 == 1)
	{
		paddedIndex--; // Begin at previous SCHI instruction
	}

	instruction->SetTargetAddress(
		paddedIndex * sizeof(std::uint64_t),
		m_index * sizeof(std::uint64_t)
	);
}

// NVIDIA custom ELF

void Assembler::Visit(SASS::EXITInstruction *instruction)
{
	m_binaryFunction->AddExitOffset(m_index * sizeof(std::uint64_t));
}

void Assembler::Visit(SASS::S2RInstruction *instruction)
{
	auto kind = instruction->GetSource()->GetKind();
	if (kind == SASS::SpecialRegister::Kind::SR_CTAID_X ||
		kind == SASS::SpecialRegister::Kind::SR_CTAID_Y ||
		kind == SASS::SpecialRegister::Kind::SR_CTAID_X)
	{
		m_binaryFunction->AddS2RCTAIDOffset(m_index * sizeof(std::uint64_t));
	}
	if (kind == SASS::SpecialRegister::Kind::SR_CTAID_Z)
	{
		m_binaryFunction->SetCTAIDZUsed(true);
	}
}

void Assembler::Visit(SASS::SHFLInstruction *instruction)
{
	m_binaryFunction->AddCoopOffset(m_index * sizeof(std::uint64_t));
}

void Assembler::Visit(SASS::BARInstruction *instruction)
{
	auto barrier = instruction->GetBarrier();
	if (dynamic_cast<const SASS::Register *>(barrier))
	{
		m_barrierCount = SASS::MAX_BARRIERS; // Max # barriers when register is used (dynamic)
	}
	else if (auto immediateBarrier = dynamic_cast<const SASS::I32Immediate *>(barrier))
	{
		auto value = immediateBarrier->GetValue();
		if (value + 1 > m_barrierCount)
		{
			m_barrierCount = value + 1;
		}
	}
}

}
