#include "Backend/Codegen/Generators/Instructions/Synchronization/BarrierGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void BarrierGenerator::Generate(const PTX::BarrierInstruction *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);
	
	ArchitectureDispatch::Dispatch(*this, instruction);
}

void BarrierGenerator::GenerateMaxwell(const PTX::BarrierInstruction *instruction)
{
	GenerateInstruction<
		SASS::Maxwell::BARInstruction, SASS::Maxwell::MEMBARInstruction
	>(instruction);
}

void BarrierGenerator::GenerateVolta(const PTX::BarrierInstruction *instruction)
{
	GenerateInstruction<
		SASS::Volta::BARInstruction, SASS::Volta::MEMBARInstruction
	>(instruction);
}

template<class BARInstruction, class MEMBARInstruction>
void BarrierGenerator::GenerateInstruction(const PTX::BarrierInstruction *instruction)
{
	// Generate barrier operand

	CompositeGenerator compositeGenerator(this->m_builder);
	compositeGenerator.SetZeroRegister(false);
	compositeGenerator.SetImmediateSize(4);

	auto barrier = compositeGenerator.Generate(instruction->GetBarrier());

	// Mode

	auto mode = (instruction->GetWait()) ? BARInstruction::Mode::SYNC : BARInstruction::Mode::ARV;

	// Generate instruction, checking for threads (optional)

	if (auto threadsOperand = instruction->GetThreads())
	{
		compositeGenerator.SetImmediateSize(12);
		auto threads = compositeGenerator.Generate(threadsOperand);

		// If both threads and barrier are stored in registers, they must be merged into a single register

		if (barrier->GetKind() == SASS::Operand::Kind::Register && threads->GetKind() == SASS::Operand::Kind::Register)
		{
			Error(instruction, "unsupported barrier and threads both register");
		}

		this->AddInstruction(new BARInstruction(mode, barrier, threads));
	}
	else
	{
		this->AddInstruction(new BARInstruction(mode, barrier));
	}
	this->AddInstruction(new MEMBARInstruction(MEMBARInstruction::Scope::CTA));
}

}
}
