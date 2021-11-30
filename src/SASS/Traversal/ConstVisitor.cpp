#include "SASS/Traversal/ConstVisitor.h"

#include "SASS/Tree/Tree.h"

namespace SASS {

// Structure

void ConstVisitor::Visit(const Node *node)
{

}

void ConstVisitor::Visit(const Program *program)
{
	Visit(static_cast<const Node *>(program));
}

void ConstVisitor::Visit(const Function *function)
{
	Visit(static_cast<const Node *>(function));
}

void ConstVisitor::Visit(const BasicBlock *block)
{
	Visit(static_cast<const Node *>(block));
}

void ConstVisitor::Visit(const Variable *variable)
{
	Visit(static_cast<const Node *>(variable));
}

void ConstVisitor::Visit(const GlobalVariable *variable)
{
	Visit(static_cast<const Variable *>(variable));
}

void ConstVisitor::Visit(const SharedVariable *variable)
{
	Visit(static_cast<const Variable *>(variable));
}

void ConstVisitor::Visit(const DynamicSharedVariable *variable)
{
	Visit(static_cast<const Node *>(variable));
}

void ConstVisitor::Visit(const Relocation *relocation)
{
	Visit(static_cast<const Node *>(relocation));
}

void ConstVisitor::Visit(const IndirectBranch *branch)
{
	Visit(static_cast<const Node *>(branch));
}

// Instructions

void ConstVisitor::Visit(const Instruction *instruction)
{
	Visit(static_cast<const Node *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::Instruction *instruction)
{
	Visit(static_cast<const SASS::Instruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::PredicatedInstruction *instruction)
{
	Visit(static_cast<const Maxwell::Instruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::SCHIInstruction *instruction)
{
	Visit(static_cast<const Maxwell::Instruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::Instruction *instruction)
{
	Visit(static_cast<const SASS::Instruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::PredicatedInstruction *instruction)
{
	Visit(static_cast<const Volta::Instruction *>(instruction));
}

// Control

void ConstVisitor::Visit(const Maxwell::DivergenceInstruction *instruction)
{
	Visit(static_cast<const Maxwell::Instruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::SSYInstruction *instruction)
{
	Visit(static_cast<const Maxwell::DivergenceInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::PBKInstruction *instruction)
{
	Visit(static_cast<const Maxwell::DivergenceInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::PCNTInstruction *instruction)
{
	Visit(static_cast<const Maxwell::DivergenceInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::SYNCInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::BRKInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::CONTInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::BRAInstruction *instruction)
{
	Visit(static_cast<const Maxwell::DivergenceInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::EXITInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::RETInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::ControlInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::DivergenceInstruction *instruction)
{
	Visit(static_cast<const Volta::ControlInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::BRAInstruction *instruction)
{
	Visit(static_cast<const Volta::DivergenceInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::EXITInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::BSSYInstruction *instruction)
{
	Visit(static_cast<const Volta::DivergenceInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::BSYNCInstruction *instruction)
{
	Visit(static_cast<const Volta::ControlInstruction *>(instruction));
}

// Conversion

void ConstVisitor::Visit(const Maxwell::F2FInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::F2IInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::I2FInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::I2IInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::F2FInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::F2IInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::I2FInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::I2IInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::FRNDInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}
// Integer

void ConstVisitor::Visit(const Maxwell::BFEInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::FLOInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::IADDInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::IADD3Instruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::IADD32IInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::ISCADDInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::ISETPInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::LOPInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::LOP32IInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::POPCInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::SHLInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::SHRInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::XMADInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::FLOInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::IABSInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::IADD3Instruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::IMADInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::ISETPInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::LOP3Instruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::SHFInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

// Float

void ConstVisitor::Visit(const Maxwell::DADDInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::DFMAInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::DMNMXInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::DMULInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::DSETPInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::FADDInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::MUFUInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::DADDInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::DFMAInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::DMULInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::DSETPInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::FADDInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::MUFUInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

// LoadStore

void ConstVisitor::Visit(const Maxwell::ATOMInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::ATOMCASInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::LDGInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::LDSInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::MEMBARInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::REDInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::STGInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::STSInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::ATOMGInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::LDGInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::LDSInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::MEMBARInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::REDInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::STGInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::STSInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}
// Misc

void ConstVisitor::Visit(const Maxwell::CS2RInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::DEPBARInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::BARInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::NOPInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::S2RInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::CS2RInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::DEPBARInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::BARInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::NOPInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::S2RInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

// Movement

void ConstVisitor::Visit(const Maxwell::MOVInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::MOV32IInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::PRMTInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::SELInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Maxwell::SHFLInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::MOVInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::PRMTInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::SELInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::SHFLInstruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

// Predicate

void ConstVisitor::Visit(const Maxwell::PSETPInstruction *instruction)
{
	Visit(static_cast<const Maxwell::PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const Volta::PLOP3Instruction *instruction)
{
	Visit(static_cast<const Volta::PredicatedInstruction *>(instruction));
}

// Operands

void ConstVisitor::Visit(const Operand *operand)
{
	Visit(static_cast<const Node *>(operand));
}

void ConstVisitor::Visit(const Address *address)
{
	Visit(static_cast<const Operand *>(address));
}

void ConstVisitor::Visit(const Composite *composite)
{
	Visit(static_cast<const Operand *>(composite));
}

void ConstVisitor::Visit(const Constant *constant)
{
	Visit(static_cast<const Composite *>(constant));
}

void ConstVisitor::Visit(const Immediate *immediate)
{
	Visit(static_cast<const Composite *>(immediate));
}

void ConstVisitor::Visit(const I8Immediate *immediate)
{
	Visit(static_cast<const Composite *>(immediate));
}

void ConstVisitor::Visit(const I16Immediate *immediate)
{
	Visit(static_cast<const Composite *>(immediate));
}

void ConstVisitor::Visit(const I32Immediate *immediate)
{
	Visit(static_cast<const Composite *>(immediate));
}

void ConstVisitor::Visit(const F32Immediate *immediate)
{
	Visit(static_cast<const Composite *>(immediate));
}

void ConstVisitor::Visit(const Register *reg)
{
	Visit(static_cast<const Composite *>(reg));
}

void ConstVisitor::Visit(const Predicate *reg)
{
	Visit(static_cast<const Operand *>(reg));
}

void ConstVisitor::Visit(const SpecialRegister *reg)
{
	Visit(static_cast<const Operand *>(reg));
}

void ConstVisitor::Visit(const CarryFlag *carry)
{
	Visit(static_cast<const Operand *>(carry));

}

}
