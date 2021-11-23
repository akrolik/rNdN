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

	m_computeCapability = program->GetComputeCapability();

	auto binaryProgram = new BinaryProgram();
	binaryProgram->SetComputeCapability(m_computeCapability);

	if (SASS::Maxwell::IsSupported(m_computeCapability))
	{
		m_instructionSize = sizeof(std::uint64_t);
		m_instructionPad = 6;
		m_scheduleGroup = 3;
	}
	else if (SASS::Volta::IsSupported(m_computeCapability))
	{
		m_instructionSize = 2 * sizeof(std::uint64_t);
		m_instructionPad = 16;
		m_scheduleGroup = 0;
	}
	else
	{
		Utils::Logger::LogError("Unsupported compute capability for assembler 'sm_" + std::to_string(m_computeCapability) + "'");
	}

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
	auto binaryFunction = new BinaryFunction(function->GetName(), m_instructionSize);

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
	m_blockAddress.clear();

	std::vector<const SASS::Instruction *> linearProgram;
	for (const auto block : function->GetBasicBlocks())
	{
		// Insert instructions from the basic block

		const auto& instructions = block->GetInstructions();
		linearProgram.insert(std::end(linearProgram), std::begin(instructions), std::end(instructions));

		// Keep track of the start address to resolve branches

		m_blockAddress.insert({block->GetName(), blockOffset});
		blockOffset += instructions.size() * m_instructionSize;
	}

	// Insert self-loop

	auto selfName = "_END";
	auto selfBlock = new SASS::BasicBlock(selfName);
	auto selfBranch = GenerateSelfBranchInstruction(selfName);

	linearProgram.push_back(selfBranch);
	m_blockAddress.insert({selfName, blockOffset});
	blockOffset += m_instructionSize;

	// Padding to multiple of instructions with NOPs

	const auto padding = m_instructionPad - (linearProgram.size() % m_instructionPad);

	for (auto i = 0u; i < padding; i++)
	{
		// Insert NOPs with scheduling directive:
		//   0000 000000 111 111 0 0000

		auto nop = GeneratePaddingInstruction();
		auto& nopSchedule = nop->GetSchedule();
		nopSchedule.SetStall(0);

		linearProgram.push_back(nop);
	}

	// Insert SCHI instructions for each schedule group

	if (m_scheduleGroup == 3)
	{
		// Construct SCHI instructions

		for (auto i = 0u; i < linearProgram.size(); i += 4)
		{
			auto inst1 = linearProgram.at(i);
			auto inst2 = linearProgram.at(i+1);
			auto inst3 = linearProgram.at(i+2);

			std::uint64_t assembled = 0u;
			assembled <<= 1;
			assembled |= inst3->GetSchedule().ToBinary();
			assembled <<= 21;
			assembled |= inst2->GetSchedule().ToBinary();
			assembled <<= 21;
			assembled |= inst1->GetSchedule().ToBinary();

			linearProgram.insert(std::begin(linearProgram) + i, new SASS::Maxwell::SCHIInstruction(assembled));
		}
	}
	else if (m_scheduleGroup != 0)
	{
		Utils::Logger::LogError("Unsupported schedule group size '" + std::to_string(m_scheduleGroup) + "'");
	}

	// Resolve branches and control reconvergence instructions
	// Collect special NVIDIA ELF properties

	m_currentAddress = 0;
	m_barrierCount = 0;
	m_binaryFunction = binaryFunction;

	for (auto i = 0u; i < linearProgram.size(); ++i)
	{
		auto instruction = const_cast<SASS::Instruction *>(linearProgram.at(i));
		instruction->Accept(*this);

		m_currentAddress += m_instructionSize;
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
		auto address = (it - linearProgram.begin()) * m_instructionSize;

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
			case SASS::Relocation::Kind::ABS32_LO_32:
			{
				kind = BinaryFunction::RelocationKind::ABS32_LO_32;
				break;
			}
			case SASS::Relocation::Kind::ABS32_HI_32:
			{
				kind = BinaryFunction::RelocationKind::ABS32_HI_32;
				break;
			}
			case SASS::Relocation::Kind::ABS24_20:
			{
				kind = BinaryFunction::RelocationKind::ABS24_20;
				break;
			}
			case SASS::Relocation::Kind::ABS32_32:
			{
				kind = BinaryFunction::RelocationKind::ABS32_32;
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
		auto offsetAddress = (it - linearProgram.begin())  * m_instructionSize;

		// Compute offset of target block

		auto targetAddress = GetTargetAddress(indirectBranch->GetTarget());

		// Add indirect branch

		binaryFunction->AddIndirectBranch(offsetAddress, targetAddress);
	}

	// Stack size

	binaryFunction->SetCRSStackSize(function->GetCRSStackSize());

	// Assemble into binary

	auto binaryFactor = m_instructionSize / sizeof(std::uint64_t);
	auto binary = new std::uint64_t[linearProgram.size() * binaryFactor];
	for (auto i = 0u; i < linearProgram.size(); ++i)
	{
		auto instruction = linearProgram[i];
		binary[i * binaryFactor] = instruction->ToBinary();

		if (m_instructionSize == 16)
		{
			binary[i * binaryFactor + 1] = instruction->ToBinaryHi();
		}
	}
	binaryFunction->SetText((char *)binary, m_instructionSize * linearProgram.size());
	binaryFunction->SetRegisters(function->GetRegisters());
	binaryFunction->SetMaxRegisters(function->GetMaxRegisters());
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

	binaryFunction->SetConstantMemory(function->GetConstantMemory(), function->GetConstantMemoryAlign());

	return binaryFunction;
}

// Padding instructions

SASS::Instruction *Assembler::GeneratePaddingInstruction() const
{
	if (SASS::Maxwell::IsSupported(m_computeCapability))
	{
		return new SASS::Maxwell::NOPInstruction();
	}
	else if (SASS::Volta::IsSupported(m_computeCapability))
	{
		return new SASS::Volta::NOPInstruction();
	}

	Utils::Logger::LogError("Unsupported compute capability for padding instruction '" + std::to_string(m_computeCapability) + "'");
}

SASS::Instruction *Assembler::GenerateSelfBranchInstruction(const std::string& name) const
{
	if (SASS::Maxwell::IsSupported(m_computeCapability))
	{
		auto instruction = new SASS::Maxwell::BRAInstruction(name);
		auto& schedule = instruction->GetSchedule();

		schedule.SetStall(15);
		schedule.SetYield(true);

		return instruction;
	}
	else if (SASS::Volta::IsSupported(m_computeCapability))
	{
		auto instruction = new SASS::Volta::BRAInstruction(name);
		auto& schedule = instruction->GetSchedule();
		schedule.SetStall(0);
		return instruction;
	}

	Utils::Logger::LogError("Unsupported compute capability for branch instruction '" + std::to_string(m_computeCapability) + "'");
}

// Address resolution

std::uint64_t Assembler::GetTargetAddress(const std::string& target) const
{
	auto unpaddedAddress = m_blockAddress.at(target);

	// Inline scheduling directives are not padded

	if (m_scheduleGroup == 0)
	{
		return unpaddedAddress;
	}

	// Separate SCHI instructions require padding for address resolution

	auto unpaddedIndex = unpaddedAddress / m_instructionSize;
	auto paddedIndex = 1 + unpaddedIndex + (unpaddedIndex / m_scheduleGroup);

	if (paddedIndex % (m_scheduleGroup + 1) == 1)
	{
		paddedIndex--; // Begin at previous SCHI instruction
	}

	return paddedIndex * m_instructionSize;
}

void Assembler::Visit(SASS::Maxwell::DivergenceInstruction *instruction)
{
	auto targetAddress = GetTargetAddress(instruction->GetTarget());

	instruction->SetTargetAddress(targetAddress, m_currentAddress);
}

void Assembler::Visit(SASS::Volta::DivergenceInstruction *instruction)
{
	auto targetAddress = GetTargetAddress(instruction->GetTarget());

	instruction->SetTargetAddress(targetAddress, m_currentAddress);
}

// NVIDIA custom ELF

void Assembler::Visit(SASS::Maxwell::EXITInstruction *instruction)
{
	VisitEXIT(instruction);
}

void Assembler::Visit(SASS::Maxwell::S2RInstruction *instruction)
{
	VisitS2R(instruction);
}

void Assembler::Visit(SASS::Maxwell::SHFLInstruction *instruction)
{
	VisitSHFL(instruction);
}

void Assembler::Visit(SASS::Maxwell::BARInstruction *instruction)
{
	VisitBAR(instruction);
}

void Assembler::Visit(SASS::Volta::EXITInstruction *instruction)
{
	VisitEXIT(instruction);
}

void Assembler::Visit(SASS::Volta::S2RInstruction *instruction)
{
	VisitS2R(instruction);
}

void Assembler::Visit(SASS::Volta::SHFLInstruction *instruction)
{
	VisitSHFL(instruction);
}

void Assembler::Visit(SASS::Volta::BARInstruction *instruction)
{
	VisitBAR(instruction);
}

template<class T>
void Assembler::VisitEXIT(T *instruction)
{
	m_binaryFunction->AddExitOffset(m_currentAddress);
}

template<class T>
void Assembler::VisitS2R(T *instruction)
{
	auto kind = instruction->GetSource()->GetKind();
	if (kind == SASS::SpecialRegister::Kind::SR_CTAID_X ||
		kind == SASS::SpecialRegister::Kind::SR_CTAID_Y ||
		kind == SASS::SpecialRegister::Kind::SR_CTAID_X)
	{
		m_binaryFunction->AddS2RCTAIDOffset(m_currentAddress);
	}
	if (kind == SASS::SpecialRegister::Kind::SR_CTAID_Z)
	{
		m_binaryFunction->SetCTAIDZUsed(true);
	}
}

template<class T>
void Assembler::VisitSHFL(T *instruction)
{
	m_binaryFunction->AddCoopOffset(m_currentAddress);
}

template<class T>
void Assembler::VisitBAR(T *instruction)
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
