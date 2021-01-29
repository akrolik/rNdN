#pragma once

#include "Backend/Codegen/Generators/Generator.h"
#include "PTX/Traversal/ConstInstructionVisitor.h"

#include "PTX/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class InstructionGenerator : public PTX::ConstInstructionVisitor, public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "InstructionGenerator"; }

	// Arithmetic

	// void Visit(const PTX::_AbsoluteInstruction *instruction) override;
	void Visit(const PTX::_AddInstruction *instruction) override;
	// void Visit(const PTX::_CountLeadingZerosInstruction *instruction) override;
	// void Visit(const PTX::_DivideInstruction *instruction) override;
	void Visit(const PTX::_MADInstruction *instruction) override;
	void Visit(const PTX::_MultiplyInstruction *instruction) override;
	void Visit(const PTX::_MultiplyWideInstruction *instruction) override;
	// void Visit(const PTX::_NegateInstruction *instruction) override;
	// void Visit(const PTX::_ReciprocalInstruction *instruction) override;
	void Visit(const PTX::_RemainderInstruction *instruction) override;
	// void Visit(const PTX::_SubtractInstruction *instruction) override;

	// Comparison

	void Visit(const PTX::_SelectInstruction *instruction) override;
	void Visit(const PTX::_SetPredicateInstruction *instruction) override;

	// Control Flow

	void Visit(const PTX::BranchInstruction *instruction) override;
	// void Visit(const PTX::_CallInstruction *instruction) override;
	void Visit(const PTX::ReturnInstruction *instruction) override;

	// Data 

	void Visit(const PTX::_ConvertInstruction *instruction) override;
	void Visit(const PTX::_ConvertToAddressInstruction *instruction) override;
	void Visit(const PTX::_LoadInstruction *instruction) override;
	void Visit(const PTX::_LoadNCInstruction *instruction) override;
	void Visit(const PTX::_MoveInstruction *instruction) override;
	// void Visit(const PTX::_MoveAddressInstruction *instruction) override;
	void Visit(const PTX::_MoveSpecialInstruction *instruction) override;
	void Visit(const PTX::_PackInstruction *instruction) override;
	// void Visit(const PTX::_ShuffleInstruction *instruction) override;
	void Visit(const PTX::_StoreInstruction *instruction) override;
	void Visit(const PTX::_UnpackInstruction *instruction) override;

	// Logical

	// void Visit(const PTX::_AndInstruction *instruction) override;
	// void Visit(const PTX::_NotInstruction *instruction) override;
	// void Visit(const PTX::_OrInstruction *instruction) override;
	// void Visit(const PTX::_XorInstruction *instruction) override;

	// Shift

	// void Visit(const PTX::_ShiftLeftInstruction *instruction) override;
	// void Visit(const PTX::_ShiftRightInstruction *instruction) override;

	// Synchronization

	// void Visit(const PTX::_AtomicInstruction *instruction) override;
	// void Visit(const PTX::_BarrierInstruction *instruction) override;
	// void Visit(const PTX::_ReductionInstruction *instruction) override;
};

}
}
