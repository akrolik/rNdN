#include "SASS/Traversal/Visitor.h"

#include "SASS/Tree/Tree.h"

namespace SASS {

// Structure

void Visitor::Visit(Node *node)
{

}

void Visitor::Visit(Program *program)
{
	Visit(static_cast<Node *>(program));
}

void Visitor::Visit(Function *function)
{
	Visit(static_cast<Node *>(function));
}

void Visitor::Visit(BasicBlock *block)
{
	Visit(static_cast<Node *>(block));
}

void Visitor::Visit(Variable *variable)
{
	Visit(static_cast<Node *>(variable));
}

void Visitor::Visit(GlobalVariable *variable)
{
	Visit(static_cast<Variable *>(variable));
}

void Visitor::Visit(SharedVariable *variable)
{
	Visit(static_cast<Variable *>(variable));
}

void Visitor::Visit(DynamicSharedVariable *variable)
{
	Visit(static_cast<Node *>(variable));
}

void Visitor::Visit(Relocation *relocation)
{
	Visit(static_cast<Node *>(relocation));
}

void Visitor::Visit(IndirectBranch *branch)
{
	Visit(static_cast<Node *>(branch));
}

// Instructions

void Visitor::Visit(Instruction *instruction)
{
	Visit(static_cast<Node *>(instruction));
}

void Visitor::Visit(Maxwell::Instruction *instruction)
{
	Visit(static_cast<SASS::Instruction *>(instruction));
}

void Visitor::Visit(Maxwell::PredicatedInstruction *instruction)
{
	Visit(static_cast<Maxwell::Instruction *>(instruction));
}

void Visitor::Visit(Maxwell::SCHIInstruction *instruction)
{
	Visit(static_cast<Maxwell::Instruction *>(instruction));
}

void Visitor::Visit(Volta::Instruction *instruction)
{
	Visit(static_cast<SASS::Instruction *>(instruction));
}

void Visitor::Visit(Volta::PredicatedInstruction *instruction)
{
	Visit(static_cast<Volta::Instruction *>(instruction));
}

// Control

void Visitor::Visit(Maxwell::DivergenceInstruction *instruction)
{
	Visit(static_cast<Maxwell::Instruction *>(instruction));
}

void Visitor::Visit(Maxwell::SSYInstruction *instruction)
{
	Visit(static_cast<Maxwell::DivergenceInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::PBKInstruction *instruction)
{
	Visit(static_cast<Maxwell::DivergenceInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::PCNTInstruction *instruction)
{
	Visit(static_cast<Maxwell::DivergenceInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::SYNCInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::BRKInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::CONTInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::BRAInstruction *instruction)
{
	Visit(static_cast<Maxwell::DivergenceInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::EXITInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::RETInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::ControlInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::DivergenceInstruction *instruction)
{
	Visit(static_cast<Volta::ControlInstruction *>(instruction));
}

void Visitor::Visit(Volta::BRAInstruction *instruction)
{
	Visit(static_cast<Volta::DivergenceInstruction *>(instruction));
}

void Visitor::Visit(Volta::EXITInstruction *instruction)
{
	Visit(static_cast<Volta::ControlInstruction *>(instruction));
}

void Visitor::Visit(Volta::BSSYInstruction *instruction)
{
	Visit(static_cast<Volta::DivergenceInstruction *>(instruction));
}

void Visitor::Visit(Volta::BSYNCInstruction *instruction)
{
	Visit(static_cast<Volta::ControlInstruction *>(instruction));
}

// Conversion

void Visitor::Visit(Maxwell::F2FInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::F2IInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::I2FInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::I2IInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::F2FInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::F2IInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::I2FInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::I2IInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::FRNDInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

// Integer

void Visitor::Visit(Maxwell::BFEInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::FLOInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::IADDInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::IADD3Instruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::IADD32IInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::ISCADDInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::ISETPInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::LOPInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::LOP32IInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::POPCInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::SHLInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::SHRInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::XMADInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::FLOInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::IADD3Instruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::IMADInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::ISETPInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::LOP3Instruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::SHFInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

// Float

void Visitor::Visit(Maxwell::DADDInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::DFMAInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::DMNMXInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::DMULInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::DSETPInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::FADDInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::MUFUInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::DADDInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::DFMAInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::DMULInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::DSETPInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::FADDInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::MUFUInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

// LoadStore

void Visitor::Visit(Maxwell::ATOMInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::ATOMCASInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::LDGInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::LDSInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::MEMBARInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::REDInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::STGInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::STSInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::ATOMGInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::LDGInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::LDSInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::MEMBARInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::REDInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::STGInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::STSInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

// Misc

void Visitor::Visit(Maxwell::CS2RInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::DEPBARInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::BARInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::NOPInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::S2RInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::CS2RInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::DEPBARInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::BARInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::NOPInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::S2RInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

// Movement

void Visitor::Visit(Maxwell::MOVInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::MOV32IInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::SELInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Maxwell::SHFLInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::MOVInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::SELInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::SHFLInstruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

// Predicate

void Visitor::Visit(Maxwell::PSETPInstruction *instruction)
{
	Visit(static_cast<Maxwell::PredicatedInstruction *>(instruction));
}

void Visitor::Visit(Volta::PLOP3Instruction *instruction)
{
	Visit(static_cast<Volta::PredicatedInstruction *>(instruction));
}

// Operands

void Visitor::Visit(Operand *operand)
{
	Visit(static_cast<Node *>(operand));
}

void Visitor::Visit(Address *address)
{
	Visit(static_cast<Operand *>(address));
}

void Visitor::Visit(Composite *composite)
{
	Visit(static_cast<Operand *>(composite));
}

void Visitor::Visit(Constant *constant)
{
	Visit(static_cast<Composite *>(constant));
}

void Visitor::Visit(Immediate *immediate)
{
	Visit(static_cast<Composite *>(immediate));
}

void Visitor::Visit(I8Immediate *immediate)
{
	Visit(static_cast<Composite *>(immediate));
}

void Visitor::Visit(I16Immediate *immediate)
{
	Visit(static_cast<Composite *>(immediate));
}

void Visitor::Visit(I32Immediate *immediate)
{
	Visit(static_cast<Composite *>(immediate));
}

void Visitor::Visit(F32Immediate *immediate)
{
	Visit(static_cast<Composite *>(immediate));
}

void Visitor::Visit(Register *reg)
{
	Visit(static_cast<Composite *>(reg));
}

void Visitor::Visit(Predicate *reg)
{
	Visit(static_cast<Operand *>(reg));
}

void Visitor::Visit(SpecialRegister *reg)
{
	Visit(static_cast<Operand *>(reg));
}

void Visitor::Visit(CarryFlag *carry)
{
	Visit(static_cast<Operand *>(carry));

}

}
