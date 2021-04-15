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

void Visitor::Visit(PredicatedInstruction *instruction)
{
	Visit(static_cast<Instruction *>(instruction));
}

void Visitor::Visit(SCHIInstruction *instruction)
{
	Visit(static_cast<Instruction *>(instruction));
}

// Control

void Visitor::Visit(DivergenceInstruction *instruction)
{
	Visit(static_cast<Instruction *>(instruction));
}

void Visitor::Visit(SSYInstruction *instruction)
{
	Visit(static_cast<DivergenceInstruction *>(instruction));
}

void Visitor::Visit(PBKInstruction *instruction)
{
	Visit(static_cast<DivergenceInstruction *>(instruction));
}

void Visitor::Visit(PCNTInstruction *instruction)
{
	Visit(static_cast<DivergenceInstruction *>(instruction));
}

void Visitor::Visit(SYNCInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(BRKInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(CONTInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(BRAInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(EXITInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(RETInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

// Conversion

void Visitor::Visit(F2IInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(I2FInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(I2IInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

// Integer

void Visitor::Visit(BFEInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(FLOInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(IADDInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(IADD3Instruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(IADD32IInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(ISCADDInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(ISETPInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(LOPInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(LOP32IInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(POPCInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(SHLInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(SHRInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(XMADInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

// Float

void Visitor::Visit(DADDInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(DFMAInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(DMNMXInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(DMULInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(DSETPInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(MUFUInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

// LoadStore

void Visitor::Visit(ATOMInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(LDGInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(LDSInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(MEMBARInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(REDInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(STGInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(STSInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

// Misc

void Visitor::Visit(CS2RInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(DEPBARInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(BARInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(NOPInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(S2RInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

// Movement

void Visitor::Visit(MOVInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(MOV32IInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(SELInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}

void Visitor::Visit(SHFLInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
}


// Predicate

void Visitor::Visit(PSETPInstruction *instruction)
{
	Visit(static_cast<PredicatedInstruction *>(instruction));
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
