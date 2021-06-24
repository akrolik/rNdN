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

void ConstVisitor::Visit(const PredicatedInstruction *instruction)
{
	Visit(static_cast<const Instruction *>(instruction));
}

void ConstVisitor::Visit(const SCHIInstruction *instruction)
{
	Visit(static_cast<const Instruction *>(instruction));
}

// Control

void ConstVisitor::Visit(const DivergenceInstruction *instruction)
{
	Visit(static_cast<const Instruction *>(instruction));
}

void ConstVisitor::Visit(const SSYInstruction *instruction)
{
	Visit(static_cast<const DivergenceInstruction *>(instruction));
}

void ConstVisitor::Visit(const PBKInstruction *instruction)
{
	Visit(static_cast<const DivergenceInstruction *>(instruction));
}

void ConstVisitor::Visit(const PCNTInstruction *instruction)
{
	Visit(static_cast<const DivergenceInstruction *>(instruction));
}

void ConstVisitor::Visit(const SYNCInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const BRKInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const CONTInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const BRAInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const EXITInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const RETInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

// Conversion

void ConstVisitor::Visit(const F2IInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const I2FInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const I2IInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

// Integer

void ConstVisitor::Visit(const BFEInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const FLOInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const IADDInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const IADD3Instruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const IADD32IInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const ISCADDInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const ISETPInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const LOPInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const LOP32IInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const POPCInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const SHLInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const SHRInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const XMADInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

// Float

void ConstVisitor::Visit(const DADDInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const DFMAInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const DMNMXInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const DMULInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const DSETPInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const MUFUInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

// LoadStore

void ConstVisitor::Visit(const ATOMInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const ATOMCASInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const LDGInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const LDSInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const MEMBARInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const REDInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const STGInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const STSInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

// Misc

void ConstVisitor::Visit(const CS2RInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const DEPBARInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const BARInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const NOPInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const S2RInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

// Movement

void ConstVisitor::Visit(const MOVInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const MOV32IInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const SELInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}

void ConstVisitor::Visit(const SHFLInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
}


// Predicate

void ConstVisitor::Visit(const PSETPInstruction *instruction)
{
	Visit(static_cast<const PredicatedInstruction *>(instruction));
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
