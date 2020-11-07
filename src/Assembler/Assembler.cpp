#include "Assembler/Assembler.h"

#include <cstring>
#include <unordered_map>
#include <vector>

#include "Assembler/ELFGenerator.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Assembler {

ELFBinary *Assembler::Assemble(const SASS::Program *program)
{
	auto binaryProgram = AssembleProgram(program);

	ELFGenerator elfGenerator;
	return elfGenerator.Generate(binaryProgram);
}

BinaryProgram *Assembler::AssembleProgram(const SASS::Program *program)
{
	auto timeAssembler_start = Utils::Chrono::Start("SASS assembler");

	// Collect the device properties for codegen

	auto binaryProgram = new BinaryProgram();
	binaryProgram->SetComputeCapability(program->GetComputeCapability());
	for (const auto& function : program->GetFunctions())
	{
		binaryProgram->AddFunction(AssembleFunction(function));
	}

	Utils::Chrono::End(timeAssembler_start);

	return binaryProgram;
}

BinaryFunction *Assembler::AssembleFunction(const SASS::Function *function)
{
	auto binaryFunction = new BinaryFunction(function->GetName());
	for (auto parameter : function->GetParameters())
	{
		binaryFunction->AddParameter(parameter);
	}

	std::unordered_map<const SASS::BasicBlock *, unsigned int> blockIndex;
	auto blockOffset = 0u;

	// 1. Sequence basic blocks, create linear sequence of instruction
	// 2. Add self-loop BRA
	// 3. Pad instructions to multiple of 6
	// 4. Add 1 SCHI instruction for every 3 regular instructions (now padded to multiple of 8)
	// 5. Resolve branch targets
	//    (a) BRA instructions
	//    (b) EXIT instructions
	//    (c) CTAID instructions

	std::vector<SASS::Instruction *> linearProgram;
	for (const auto block : function->GetBasicBlocks())
	{
		// Insert instructions from the basic block

		const auto& instructions = block->GetInstructions();
		linearProgram.insert(std::end(linearProgram), std::begin(instructions), std::end(instructions));

		// Keep track of the start address to resolve branches

		blockIndex.insert({block, blockOffset});
		blockOffset += instructions.size();
	}

	// Insert self-loop

	auto selfBlock = new SASS::BasicBlock("_END");
	auto selfBranch = new SASS::BRAInstruction(selfBlock);
	selfBranch->SetScheduling(15, true, 7, 7, 0, 0);

	linearProgram.push_back(selfBranch);
	blockIndex.insert({selfBlock, blockOffset++});

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

	// Resolve branches and collect special instructions for Nvidia ELF

	for (auto i = 0u; i < linearProgram.size(); ++i)
	{
		auto instruction = linearProgram.at(i);
		if (auto branchInstruction = dynamic_cast<SASS::BRAInstruction *>(instruction))
		{
			auto unpaddedIndex = blockIndex.at(branchInstruction->GetTarget());
			auto paddedIndex = 1 + unpaddedIndex + (unpaddedIndex / 3);

			branchInstruction->SetTargetAddress(
				paddedIndex * sizeof(std::uint64_t),
				i * sizeof(std::uint64_t)
			);
		}
		else if (auto exitInstruction = dynamic_cast<SASS::EXITInstruction *>(instruction))
		{
			binaryFunction->AddExitOffset(i * sizeof(std::uint64_t));
		}
		else if (auto s2rInstruction = dynamic_cast<SASS::S2RInstruction *>(instruction))
		{
			if (s2rInstruction->GetSource()->GetKind() == SASS::SpecialRegister::Kind::SR_CTAID_X)
			{
				binaryFunction->AddS2RCTAIDOffset(i * sizeof(std::uint64_t));
			}
		}
	}

	// Print assembled program with address and binary format

	//TODO: Print function metadata
	if (Utils::Options::Get<>(Utils::Options::Opt_Print_sass))
	{
		Utils::Logger::LogInfo("Assembled SASS program");
		for (auto i = 0u; i < linearProgram.size(); ++i)
		{
			auto instruction = linearProgram.at(i);

			auto address = "/* " + Utils::Format::HexString(i * sizeof(std::uint64_t), 4) + " */    ";
			auto mnemonic = instruction->ToString();
			auto binary = "/* " + Utils::Format::HexString(instruction->ToBinary(), 16) + " */";

			std::string spacing(40 - mnemonic.length(), ' ');
			Utils::Logger::LogInfo(address + mnemonic + spacing + binary, 0, true, Utils::Logger::NoPrefix);
		}
	}

	// Assemble into binary

	auto binary = new std::uint64_t[linearProgram.size()];
	for (auto i = 0u; i < linearProgram.size(); ++i)
	{
		binary[i] = linearProgram.at(i)->ToBinary();
	}
	binaryFunction->SetText((char *)binary, sizeof(std::uint64_t) * linearProgram.size());
	binaryFunction->SetRegisters(function->GetRegisters());

	return binaryFunction;
}

}
