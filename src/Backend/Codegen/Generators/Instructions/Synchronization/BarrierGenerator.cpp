#include "Backend/Codegen/Generators/Instructions/Synchronization/BarrierGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"

namespace Backend {
namespace Codegen {

void BarrierGenerator::Generate(const PTX::BarrierInstruction *instruction)
{
	// Generate barrier operand

	CompositeGenerator compositeGenerator(this->m_builder);
	compositeGenerator.SetZeroRegister(false);

	auto [barrier, barrier_Hi] = compositeGenerator.Generate(instruction->GetBarrier());

	// Mode

	auto mode = (instruction->GetWait()) ? SASS::BARInstruction::Mode::SYNC : SASS::BARInstruction::Mode::ARV;

	// Generate instruction, checking for threads (optional)

	if (auto threadsOperand = instruction->GetThreads())
	{
		auto [threads, threads_Hi] = compositeGenerator.Generate(threadsOperand);

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
