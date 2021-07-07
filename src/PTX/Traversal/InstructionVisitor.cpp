#include "PTX/Traversal/InstructionVisitor.h"

#include "PTX/Tree/Tree.h"

namespace PTX {

void InstructionVisitor::Visit(DispatchBase *instruction)
{
}

void InstructionVisitor::Visit(DevInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

// Arithmetic

void InstructionVisitor::Visit(_AbsoluteInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_AddInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_BitFieldExtractInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_BitFieldInsertInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_BitFindInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_BitReverseInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_CopySignInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_CosineInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_CountLeadingZerosInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_DivideInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_Exp2Instruction *instruction)
{
}

void InstructionVisitor::Visit(_FindNthBitInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_FMAInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_Log2Instruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_MADInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_MADWideInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_MaximumInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_MultiplyInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_MultiplyWideInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_MinimumInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_NegateInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_PopulationCountInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_ReciprocalInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_ReciprocalRootInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_RemainderInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_RootInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_SADInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_SineInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_SubtractInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_TestPropertyInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

// Comparison

void InstructionVisitor::Visit(_SelectInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_SetInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_SetPredicateInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_SignSelectInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

// Control Flow

void InstructionVisitor::Visit(BranchInstruction *instruction) // Untyped
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(BranchIndexInstruction *instruction) // Untyped
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_CallInstructionBase *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(ExitInstruction *instruction) // Untyped
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(ReturnInstruction *instruction) // Untyped
{
	Visit(static_cast<DispatchBase *>(instruction));
}

// Data

void InstructionVisitor::Visit(_ConvertAddressInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_ConvertToAddressInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_ConvertInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_IsSpaceInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_LoadInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_LoadNCInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_LoadUniformInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_MoveInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_MoveAddressInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_MoveSpecialInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_PackInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_PermuteInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_PrefetchInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_PrefetchUniformInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_ShuffleInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_StoreInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_UnpackInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

// Logical

void InstructionVisitor::Visit(_AndInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_CNotInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_Logical3OpInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_NotInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_OrInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_XorInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

// Synchronization

void InstructionVisitor::Visit(_ActiveMaskInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_AtomicInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(BarrierInstruction *instruction) // Untyped
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_BarrierReductionInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(BarrierWarpInstruction *instruction) // Untyped
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(FenceInstruction *instruction) // Untyped
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_MatchAllInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_MatchAnyInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(MemoryBarrierInstruction *instruction) // Untyped
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_ReductionInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_VoteInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

// Shift

void InstructionVisitor::Visit(_FunnelShiftInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_ShiftLeftInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

void InstructionVisitor::Visit(_ShiftRightInstruction *instruction)
{
	Visit(static_cast<DispatchBase *>(instruction));
}

}
