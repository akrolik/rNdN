#include "PTX/Traversal/ConstInstructionVisitor.h"

#include "PTX/Tree/Tree.h"

namespace PTX {

void ConstInstructionVisitor::Visit(const DispatchBase *instruction)
{
}

void ConstInstructionVisitor::Visit(const DevInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

// Arithmetic

void ConstInstructionVisitor::Visit(const _AbsoluteInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _AddInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _BitFieldExtractInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _BitFieldInsertInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _BitFindInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _BitReverseInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _CopySignInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _CosineInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _CountLeadingZerosInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _DivideInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _Exp2Instruction *instruction)
{
}

void ConstInstructionVisitor::Visit(const _FindNthBitInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _FMAInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _Log2Instruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _MADInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _MADWideInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _MaximumInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _MultiplyInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _MultiplyWideInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _MinimumInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _NegateInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _PopulationCountInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _ReciprocalInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _ReciprocalRootInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _RemainderInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _RootInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _SADInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _SineInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _SubtractInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _TestPropertyInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

// Comparison

void ConstInstructionVisitor::Visit(const _SelectInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _SetInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _SetPredicateInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _SignSelectInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

// Control Flow

void ConstInstructionVisitor::Visit(const BranchInstruction *instruction) // Untyped
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const BranchIndexInstruction *instruction) // Untyped
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _CallInstructionBase *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const ExitInstruction *instruction) // Untyped
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const ReturnInstruction *instruction) // Untyped
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

// Data

void ConstInstructionVisitor::Visit(const _ConvertAddressInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _ConvertToAddressInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _ConvertInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _IsSpaceInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _LoadInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _LoadNCInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _LoadUniformInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _MoveInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _MoveAddressInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _MoveSpecialInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _PackInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _PermuteInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _PrefetchInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _PrefetchUniformInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _ShuffleInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _StoreInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _UnpackInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

// Logical

void ConstInstructionVisitor::Visit(const _AndInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _CNotInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _Logical3OpInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _NotInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _OrInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _XorInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

// Synchronization

void ConstInstructionVisitor::Visit(const _ActiveMaskInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _AtomicInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const BarrierInstruction *instruction) // Untyped
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _BarrierReductionInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const BarrierWarpInstruction *instruction) // Untyped
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const FenceInstruction *instruction) // Untyped
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _MatchAllInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _MatchAnyInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const MemoryBarrierInstruction *instruction) // Untyped
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _ReductionInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _VoteInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

// Shift

void ConstInstructionVisitor::Visit(const _FunnelShiftInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _ShiftLeftInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

void ConstInstructionVisitor::Visit(const _ShiftRightInstruction *instruction)
{
	Visit(static_cast<const DispatchBase *>(instruction));
}

}
