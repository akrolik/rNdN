#pragma once

namespace PTX {

class DevInstruction;

// Arithmetic

class _AbsoluteInstruction;
class _AddInstruction;
class _BitFieldExtractInstruction;
class _BitFieldInsertInstruction;
class _BitFindInstruction;
class _BitReverseInstruction;
class _CopySignInstruction;
class _CosineInstruction;
class _CountLeadingZerosInstruction;
class _DivideInstruction;
class _Exp2Instruction;
class _FindNthBitInstruction;
class _FMAInstruction;
class _Log2Instruction;
class _MADInstruction;
class _MADWideInstruction;
class _MaximumInstruction;
class _MultiplyInstruction;
class _MultiplyWideInstruction;
class _MinimumInstruction;
class _NegateInstruction;
class _PopulationCountInstruction;
class _ReciprocalInstruction;
class _ReciprocalRootInstruction;
class _RemainderInstruction;
class _RootInstruction;
class _SADInstruction;
class _SineInstruction;
class _SubtractInstruction;
class _TestPropertyInstruction;

// Comparison

class _SelectInstruction;
class _SetInstruction;
class _SetPredicateInstruction;
class _SignSelectInstruction;

// Control Flow

class BranchInstruction;
class BranchIndexInstruction;
class _CallInstructionBase;
class ExitInstruction;
class ReturnInstruction;

// Data

class _ConvertAddressInstruction;
class _ConvertToAddressInstruction;
class _ConvertInstruction;
class _IsSpaceInstruction;
class _LoadInstruction;
class _LoadNCInstruction;
class _LoadUniformInstruction;
class _MoveInstruction;
class _MoveAddressInstruction;
class _MoveSpecialInstruction;
class _PackInstruction;
class _PermuteInstruction;
class _PrefetchInstruction;
class _PrefetchUniformInstruction;
class _ShuffleInstruction;
class _StoreInstruction;
class _UnpackInstruction;

// Logical

class _AndInstruction;
class _CNotInstruction;
class _Logical3OpInstruction;
class _NotInstruction;
class _OrInstruction;
class _XorInstruction;

// Synchronization

class _ActiveMaskInstruction;
class _AtomicInstruction;
class BarrierInstruction;
class _BarrierReductionInstruction;
class BarrierWarpInstruction;
class FenceInstruction;
class _MatchInstruction;
class MemoryBarrierInstruction;
class _ReductionInstruction;
class _VoteInstruction;

// Shift

class _FunnelShiftInstruction;
class _ShiftLeftInstruction;
class _ShiftRightInstruction;

class InstructionVisitor
{
public:
	virtual void Visit(DevInstruction *instruction) {}

	// Arithmetic

	virtual void Visit(_AbsoluteInstruction *instruction) {}
	virtual void Visit(_AddInstruction *instruction) {}
	virtual void Visit(_BitFieldExtractInstruction *instruction) {}
	virtual void Visit(_BitFieldInsertInstruction *instruction) {}
	virtual void Visit(_BitFindInstruction *instruction) {}
	virtual void Visit(_BitReverseInstruction *instruction) {}
	virtual void Visit(_CopySignInstruction *instruction) {}
	virtual void Visit(_CosineInstruction *instruction) {}
	virtual void Visit(_CountLeadingZerosInstruction *instruction) {}
	virtual void Visit(_DivideInstruction *instruction) {}
	virtual void Visit(_Exp2Instruction *instruction) {}
	virtual void Visit(_FindNthBitInstruction *instruction) {}
	virtual void Visit(_FMAInstruction *instruction) {}
	virtual void Visit(_Log2Instruction *instruction) {}
	virtual void Visit(_MADInstruction *instruction) {}
	virtual void Visit(_MADWideInstruction *instruction) {}
	virtual void Visit(_MaximumInstruction *instruction) {}
	virtual void Visit(_MultiplyInstruction *instruction) {}
	virtual void Visit(_MultiplyWideInstruction *instruction) {}
	virtual void Visit(_MinimumInstruction *instruction) {}
	virtual void Visit(_NegateInstruction *instruction) {}
	virtual void Visit(_PopulationCountInstruction *instruction) {}
	virtual void Visit(_ReciprocalInstruction *instruction) {}
	virtual void Visit(_ReciprocalRootInstruction *instruction) {}
	virtual void Visit(_RemainderInstruction *instruction) {}
	virtual void Visit(_RootInstruction *instruction) {}
	virtual void Visit(_SADInstruction *instruction) {}
	virtual void Visit(_SineInstruction *instruction) {}
	virtual void Visit(_SubtractInstruction *instruction) {}
        virtual void Visit(_TestPropertyInstruction *instruction) {}
 
	// Comparison

	virtual void Visit(_SelectInstruction *instruction) {}
	virtual void Visit(_SetInstruction *instruction) {}
	virtual void Visit(_SetPredicateInstruction *instruction) {}
	virtual void Visit(_SignSelectInstruction *instruction) {}

	// Control Flow

	virtual void Visit(BranchInstruction *instruction) {} // Untyped
	virtual void Visit(BranchIndexInstruction *instruction) {} // Untyped
	virtual void Visit(_CallInstructionBase *instruction) {}
	virtual void Visit(ExitInstruction *instruction) {} // Untyped
	virtual void Visit(ReturnInstruction *instruction) {} // Untyped

	// Data

	virtual void Visit(_ConvertAddressInstruction *instruction) {}
	virtual void Visit(_ConvertToAddressInstruction *instruction) {}
	virtual void Visit(_ConvertInstruction *instruction) {}
	virtual void Visit(_IsSpaceInstruction *instruction) {}
	virtual void Visit(_LoadInstruction *instruction) {}
	virtual void Visit(_LoadNCInstruction *instruction) {}
	virtual void Visit(_LoadUniformInstruction *instruction) {}
	virtual void Visit(_MoveInstruction *instruction) {}
	virtual void Visit(_MoveAddressInstruction *instruction) {}
	virtual void Visit(_MoveSpecialInstruction *instruction) {}
	virtual void Visit(_PackInstruction *instruction) {}
	virtual void Visit(_PermuteInstruction *instruction) {}
	virtual void Visit(_PrefetchInstruction *instruction) {}
	virtual void Visit(_PrefetchUniformInstruction *instruction) {}
	virtual void Visit(_ShuffleInstruction *instruction) {}
	virtual void Visit(_StoreInstruction *instruction) {}
	virtual void Visit(_UnpackInstruction *instruction) {}

	// Logical

	virtual void Visit(_AndInstruction *instruction) {}
	virtual void Visit(_CNotInstruction *instruction) {}
	virtual void Visit(_Logical3OpInstruction *instruction) {}
	virtual void Visit(_NotInstruction *instruction) {}
	virtual void Visit(_OrInstruction *instruction) {}
	virtual void Visit(_XorInstruction *instruction) {}

	// Synchronization

	virtual void Visit(_ActiveMaskInstruction *instruction) {}
	virtual void Visit(_AtomicInstruction *instruction) {}
	virtual void Visit(BarrierInstruction *instruction) {} // Untyped
	virtual void Visit(_BarrierReductionInstruction *instruction) {}
	virtual void Visit(BarrierWarpInstruction *instruction) {} // Untyped
	virtual void Visit(FenceInstruction *instruction) {} // Untyped
	virtual void Visit(_MatchInstruction *instruction) {}
	virtual void Visit(MemoryBarrierInstruction *instruction) {} // Untyped
	virtual void Visit(_ReductionInstruction *instruction) {}
	virtual void Visit(_VoteInstruction *instruction) {}

	// Shift

	virtual void Visit(_FunnelShiftInstruction *instruction) {}
	virtual void Visit(_ShiftLeftInstruction *instruction) {}
	virtual void Visit(_ShiftRightInstruction *instruction) {}
};

}
