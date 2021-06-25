#include "Backend/Codegen/Generators/Instructions/Synchronization/BarrierGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"

namespace Backend {
namespace Codegen {

void BarrierGenerator::Generate(const PTX::BarrierInstruction *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Generate barrier operand

	CompositeGenerator compositeGenerator(this->m_builder);
	compositeGenerator.SetZeroRegister(false);
	compositeGenerator.SetImmediateSize(4);

	auto barrier = compositeGenerator.Generate(instruction->GetBarrier());

	// Mode

	auto mode = (instruction->GetWait()) ? SASS::BARInstruction::Mode::SYNC : SASS::BARInstruction::Mode::ARV;

	// Generate instruction, checking for threads (optional)

	if (auto threadsOperand = instruction->GetThreads())
	{
		compositeGenerator.SetImmediateSize(12);
		auto threads = compositeGenerator.Generate(threadsOperand);

		this->AddInstruction(new SASS::BARInstruction(mode, barrier, threads));
	}
	else
	{
		this->AddInstruction(new SASS::BARInstruction(mode, barrier));
	}
	this->AddInstruction(new SASS::MEMBARInstruction(SASS::MEMBARInstruction::Mode::CTA));
}

}
}
