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

class ConstInstructionVisitor
{
public:
	virtual void Visit(const DevInstruction *instruction) {}

	// Arithmetic

	virtual void Visit(const _AbsoluteInstruction *instruction) {}
	virtual void Visit(const _AddInstruction *instruction) {}
	virtual void Visit(const _BitFieldExtractInstruction *instruction) {}
	virtual void Visit(const _BitFieldInsertInstruction *instruction) {}
	virtual void Visit(const _BitFindInstruction *instruction) {}
	virtual void Visit(const _BitReverseInstruction *instruction) {}
	virtual void Visit(const _CopySignInstruction *instruction) {}
	virtual void Visit(const _CosineInstruction *instruction) {}
	virtual void Visit(const _CountLeadingZerosInstruction *instruction) {}
	virtual void Visit(const _DivideInstruction *instruction) {}
	virtual void Visit(const _Exp2Instruction *instruction) {}
	virtual void Visit(const _FindNthBitInstruction *instruction) {}
	virtual void Visit(const _FMAInstruction *instruction) {}
	virtual void Visit(const _Log2Instruction *instruction) {}
	virtual void Visit(const _MADInstruction *instruction) {}
	virtual void Visit(const _MADWideInstruction *instruction) {}
	virtual void Visit(const _MaximumInstruction *instruction) {}
	virtual void Visit(const _MultiplyInstruction *instruction) {}
	virtual void Visit(const _MultiplyWideInstruction *instruction) {}
	virtual void Visit(const _MinimumInstruction *instruction) {}
	virtual void Visit(const _NegateInstruction *instruction) {}
	virtual void Visit(const _PopulationCountInstruction *instruction) {}
	virtual void Visit(const _ReciprocalInstruction *instruction) {}
	virtual void Visit(const _ReciprocalRootInstruction *instruction) {}
	virtual void Visit(const _RemainderInstruction *instruction) {}
	virtual void Visit(const _RootInstruction *instruction) {}
	virtual void Visit(const _SADInstruction *instruction) {}
	virtual void Visit(const _SineInstruction *instruction) {}
	virtual void Visit(const _SubtractInstruction *instruction) {}
        virtual void Visit(const _TestPropertyInstruction *instruction) {}
 
	// Comparison

	virtual void Visit(const _SelectInstruction *instruction) {}
	virtual void Visit(const _SetInstruction *instruction) {}
	virtual void Visit(const _SetPredicateInstruction *instruction) {}
	virtual void Visit(const _SignSelectInstruction *instruction) {}

	// Control Flow

	virtual void Visit(const BranchInstruction *instruction) {} // Untyped
	virtual void Visit(const BranchIndexInstruction *instruction) {} // Untyped
	virtual void Visit(const _CallInstructionBase *instruction) {}
	virtual void Visit(const ExitInstruction *instruction) {} // Untyped
	virtual void Visit(const ReturnInstruction *instruction) {} // Untyped

	// Data

	virtual void Visit(const _ConvertAddressInstruction *instruction) {}
	virtual void Visit(const _ConvertToAddressInstruction *instruction) {}
	virtual void Visit(const _ConvertInstruction *instruction) {}
	virtual void Visit(const _IsSpaceInstruction *instruction) {}
	virtual void Visit(const _LoadInstruction *instruction) {}
	virtual void Visit(const _LoadNCInstruction *instruction) {}
	virtual void Visit(const _LoadUniformInstruction *instruction) {}
	virtual void Visit(const _MoveInstruction *instruction) {}
	virtual void Visit(const _MoveAddressInstruction *instruction) {}
	virtual void Visit(const _MoveSpecialInstruction *instruction) {}
	virtual void Visit(const _PackInstruction *instruction) {}
	virtual void Visit(const _PermuteInstruction *instruction) {}
	virtual void Visit(const _PrefetchInstruction *instruction) {}
	virtual void Visit(const _PrefetchUniformInstruction *instruction) {}
	virtual void Visit(const _ShuffleInstruction *instruction) {}
	virtual void Visit(const _StoreInstruction *instruction) {}
	virtual void Visit(const _UnpackInstruction *instruction) {}

	// Logical

	virtual void Visit(const _AndInstruction *instruction) {}
	virtual void Visit(const _CNotInstruction *instruction) {}
	virtual void Visit(const _Logical3OpInstruction *instruction) {}
	virtual void Visit(const _NotInstruction *instruction) {}
	virtual void Visit(const _OrInstruction *instruction) {}
	virtual void Visit(const _XorInstruction *instruction) {}

	// Synchronization

	virtual void Visit(const _ActiveMaskInstruction *instruction) {}
	virtual void Visit(const _AtomicInstruction *instruction) {}
	virtual void Visit(const BarrierInstruction *instruction) {} // Untyped
	virtual void Visit(const _BarrierReductionInstruction *instruction) {}
	virtual void Visit(const BarrierWarpInstruction *instruction) {} // Untyped
	virtual void Visit(const FenceInstruction *instruction) {} // Untyped
	virtual void Visit(const _MatchInstruction *instruction) {}
	virtual void Visit(const MemoryBarrierInstruction *instruction) {} // Untyped
	virtual void Visit(const _ReductionInstruction *instruction) {}
	virtual void Visit(const _VoteInstruction *instruction) {}

	// Shift

	virtual void Visit(const _FunnelShiftInstruction *instruction) {}
	virtual void Visit(const _ShiftLeftInstruction *instruction) {}
	virtual void Visit(const _ShiftRightInstruction *instruction) {}
};

}
