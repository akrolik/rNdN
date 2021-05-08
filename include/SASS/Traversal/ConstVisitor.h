#pragma once

namespace SASS {

// Structure

class Node;
class Program;
class Function;
class BasicBlock;

class Variable;
class GlobalVariable;
class SharedVariable;
class DynamicSharedVariable;

class Relocation;
class IndirectBranch;

// Instructions

class Instruction;
class PredicatedInstruction;
class SCHIInstruction;

// Control

class DivergenceInstruction;

class BRAInstruction;
class BRKInstruction;
class CONTInstruction;
class EXITInstruction;
class PBKInstruction;
class PCNTInstruction;
class RETInstruction;
class SSYInstruction;
class SYNCInstruction;

// Conversion

class F2IInstruction;
class I2FInstruction;
class I2IInstruction;

// Integer

class BFEInstruction;
class FLOInstruction;
class IADDInstruction;
class IADD3Instruction;
class IADD32IInstruction;
class ISCADDInstruction;
class ISETPInstruction;
class LOPInstruction;
class LOP32IInstruction;
class POPCInstruction;
class SHLInstruction;
class SHRInstruction;
class XMADInstruction;

// Float

class DADDInstruction;
class DFMAInstruction;
class DMNMXInstruction;
class DMULInstruction;
class DSETPInstruction;
class MUFUInstruction;

// LoadStore

class ATOMInstruction;
class LDGInstruction;
class LDSInstruction;
class MEMBARInstruction;
class REDInstruction;
class STGInstruction;
class STSInstruction;

// Misc

class CS2RInstruction;
class DEPBARInstruction;
class BARInstruction;
class NOPInstruction;
class S2RInstruction;

// Movement

class MOVInstruction;
class MOV32IInstruction;
class SELInstruction;
class SHFLInstruction;

// Predicate

class PSETPInstruction;

// Operands

class Operand;

class Address;
class Composite;
class Constant;

class Immediate;
class I8Immediate;
class I16Immediate;
class I32Immediate;
class F32Immediate;

class Register;
class Predicate;
class SpecialRegister;
class CarryFlag;

class ConstVisitor
{
public:
	// Structure

	virtual void Visit(const Node *node);
	virtual void Visit(const Program *program);
	virtual void Visit(const Function *function);
	virtual void Visit(const BasicBlock *block);

	virtual void Visit(const Variable *variable);
	virtual void Visit(const GlobalVariable *variable);
	virtual void Visit(const SharedVariable *variable);
	virtual void Visit(const DynamicSharedVariable *variable);

	virtual void Visit(const Relocation *relocation);
	virtual void Visit(const IndirectBranch *branch);

	// Instructions

	virtual void Visit(const Instruction *instruction);
	virtual void Visit(const PredicatedInstruction *instruction);
	virtual void Visit(const SCHIInstruction *instruction);

	// Control

	virtual void Visit(const DivergenceInstruction *instruction);

	virtual void Visit(const SSYInstruction *instruction);
	virtual void Visit(const PBKInstruction *instruction);
	virtual void Visit(const PCNTInstruction *instruction);

	virtual void Visit(const SYNCInstruction *instruction);
	virtual void Visit(const BRKInstruction *instruction);
	virtual void Visit(const CONTInstruction *instruction);

	virtual void Visit(const BRAInstruction *instruction);
	virtual void Visit(const EXITInstruction *instruction);
	virtual void Visit(const RETInstruction *instruction);

	// Conversion

	virtual void Visit(const F2IInstruction *instruction);
	virtual void Visit(const I2FInstruction *instruction);
	virtual void Visit(const I2IInstruction *instruction);

	// Integer

	virtual void Visit(const BFEInstruction *instruction);
	virtual void Visit(const FLOInstruction *instruction);
	virtual void Visit(const IADDInstruction *instruction);
	virtual void Visit(const IADD3Instruction *instruction);
	virtual void Visit(const IADD32IInstruction *instruction);
	virtual void Visit(const ISCADDInstruction *instruction);
	virtual void Visit(const ISETPInstruction *instruction);
	virtual void Visit(const LOPInstruction *instruction);
	virtual void Visit(const LOP32IInstruction *instruction);
	virtual void Visit(const POPCInstruction *instruction);
	virtual void Visit(const SHLInstruction *instruction);
	virtual void Visit(const SHRInstruction *instruction);
	virtual void Visit(const XMADInstruction *instruction);

	// Float

	virtual void Visit(const DADDInstruction *instruction);
	virtual void Visit(const DFMAInstruction *instruction);
	virtual void Visit(const DMNMXInstruction *instruction);
	virtual void Visit(const DMULInstruction *instruction);
	virtual void Visit(const DSETPInstruction *instruction);
	virtual void Visit(const MUFUInstruction *instruction);

	// LoadStore

	virtual void Visit(const ATOMInstruction *instruction);
	virtual void Visit(const LDGInstruction *instruction);
	virtual void Visit(const LDSInstruction *instruction);
	virtual void Visit(const MEMBARInstruction *instruction);
	virtual void Visit(const REDInstruction *instruction);
	virtual void Visit(const STGInstruction *instruction);
	virtual void Visit(const STSInstruction *instruction);

	// Misc

	virtual void Visit(const CS2RInstruction *instruction);
	virtual void Visit(const DEPBARInstruction *instruction);
	virtual void Visit(const BARInstruction *instruction);
	virtual void Visit(const NOPInstruction *instruction);
	virtual void Visit(const S2RInstruction *instruction);

	// Movement

	virtual void Visit(const MOVInstruction *instruction);
	virtual void Visit(const MOV32IInstruction *instruction);
	virtual void Visit(const SELInstruction *instruction);
	virtual void Visit(const SHFLInstruction *instruction);

	// Predicate

	virtual void Visit(const PSETPInstruction *instruction);

	// Operands

	virtual void Visit(const Operand *operand);

	virtual void Visit(const Address *address);
	virtual void Visit(const Composite *composite);
	virtual void Visit(const Constant *constant);

	virtual void Visit(const Immediate *immediate);
	virtual void Visit(const I8Immediate *immediate);
	virtual void Visit(const I16Immediate *immediate);
	virtual void Visit(const I32Immediate *immediate);
	virtual void Visit(const F32Immediate *immediate);

	virtual void Visit(const Register *reg);
	virtual void Visit(const Predicate *reg);
	virtual void Visit(const SpecialRegister *reg);
	virtual void Visit(const CarryFlag *carry);
};

}
