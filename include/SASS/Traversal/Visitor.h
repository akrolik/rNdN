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

namespace Maxwell {
	class Instruction;
	class PredicatedInstruction;
	class SCHIInstruction;
}

namespace Volta {
	class Instruction;
	class PredicatedInstruction;
}

// Control

namespace Maxwell {
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
}

namespace Volta {
	class ControlInstruction;
	class DivergenceInstruction;
	class BRAInstruction;
	class EXITInstruction;
	class BSSYInstruction;
	class BSYNCInstruction;
}

// Conversion

namespace Maxwell {
	class F2FInstruction;
	class F2IInstruction;
	class I2FInstruction;
	class I2IInstruction;
}

namespace Volta {
	class F2FInstruction;
	class F2IInstruction;
	class I2FInstruction;
	class I2IInstruction;
	class FRNDInstruction;
}

// Integer

namespace Maxwell {
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
}

namespace Volta {
	class FLOInstruction;
	class IABSInstruction;
	class IADD3Instruction;
	class IMADInstruction;
	class ISETPInstruction;
	class LOP3Instruction;
	class SHFInstruction;
}

// Float

namespace Maxwell {
	class DADDInstruction;
	class DFMAInstruction;
	class DMNMXInstruction;
	class DMULInstruction;
	class DSETPInstruction;
	class FADDInstruction;
	class MUFUInstruction;
}

namespace Volta {
	class DADDInstruction;
	class DFMAInstruction;
	class DMULInstruction;
	class DSETPInstruction;
	class FADDInstruction;
	class MUFUInstruction;
}

// LoadStore

namespace Maxwell {
	class ATOMInstruction;
	class ATOMCASInstruction;
	class LDGInstruction;
	class LDSInstruction;
	class MEMBARInstruction;
	class REDInstruction;
	class STGInstruction;
	class STSInstruction;
}

namespace Volta {
	class ATOMGInstruction;
	class LDGInstruction;
	class LDSInstruction;
	class MEMBARInstruction;
	class REDInstruction;
	class STGInstruction;
	class STSInstruction;
}

// Misc

namespace Maxwell {
	class CS2RInstruction;
	class DEPBARInstruction;
	class BARInstruction;
	class NOPInstruction;
	class S2RInstruction;
}

namespace Volta {
	class CS2RInstruction;
	class DEPBARInstruction;
	class BARInstruction;
	class NOPInstruction;
	class S2RInstruction;
}

// Movement

namespace Maxwell {
	class MOVInstruction;
	class MOV32IInstruction;
	class PRMTInstruction;
	class SELInstruction;
	class SHFLInstruction;
}

namespace Volta {
	class MOVInstruction;
	class PRMTInstruction;
	class SELInstruction;
	class SHFLInstruction;
}

// Predicate

namespace Maxwell {
	class PSETPInstruction;
}

namespace Volta {
	class PLOP3Instruction;
}

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

class Visitor
{
public:
	// Structure

	virtual void Visit(Node *node);
	virtual void Visit(Program *program);
	virtual void Visit(Function *function);
	virtual void Visit(BasicBlock *block);

	virtual void Visit(Variable *variable);
	virtual void Visit(GlobalVariable *variable);
	virtual void Visit(SharedVariable *variable);
	virtual void Visit(DynamicSharedVariable *variable);

	virtual void Visit(Relocation *relocation);
	virtual void Visit(IndirectBranch *branch);

	// Instructions

	virtual void Visit(Instruction *instruction);

	virtual void Visit(Maxwell::Instruction *instruction);
	virtual void Visit(Maxwell::PredicatedInstruction *instruction);
	virtual void Visit(Maxwell::SCHIInstruction *instruction);

	virtual void Visit(Volta::Instruction *instruction);
	virtual void Visit(Volta::PredicatedInstruction *instruction);

	// Control

	virtual void Visit(Maxwell::DivergenceInstruction *instruction);

	virtual void Visit(Maxwell::SSYInstruction *instruction);
	virtual void Visit(Maxwell::PBKInstruction *instruction);
	virtual void Visit(Maxwell::PCNTInstruction *instruction);

	virtual void Visit(Maxwell::SYNCInstruction *instruction);
	virtual void Visit(Maxwell::BRKInstruction *instruction);
	virtual void Visit(Maxwell::CONTInstruction *instruction);

	virtual void Visit(Maxwell::BRAInstruction *instruction);
	virtual void Visit(Maxwell::EXITInstruction *instruction);
	virtual void Visit(Maxwell::RETInstruction *instruction);

	virtual void Visit(Volta::ControlInstruction *instruction);
	virtual void Visit(Volta::DivergenceInstruction *instruction);

	virtual void Visit(Volta::BRAInstruction *instruction);
	virtual void Visit(Volta::EXITInstruction *instruction);
	virtual void Visit(Volta::BSSYInstruction *instruction);
	virtual void Visit(Volta::BSYNCInstruction *instruction);

	// Conversion

	virtual void Visit(Maxwell::F2FInstruction *instruction);
	virtual void Visit(Maxwell::F2IInstruction *instruction);
	virtual void Visit(Maxwell::I2FInstruction *instruction);
	virtual void Visit(Maxwell::I2IInstruction *instruction);

	virtual void Visit(Volta::F2FInstruction *instruction);
	virtual void Visit(Volta::F2IInstruction *instruction);
	virtual void Visit(Volta::I2FInstruction *instruction);
	virtual void Visit(Volta::I2IInstruction *instruction);
	virtual void Visit(Volta::FRNDInstruction *instruction);

	// Integer

	virtual void Visit(Maxwell::BFEInstruction *instruction);
	virtual void Visit(Maxwell::FLOInstruction *instruction);
	virtual void Visit(Maxwell::IADDInstruction *instruction);
	virtual void Visit(Maxwell::IADD3Instruction *instruction);
	virtual void Visit(Maxwell::IADD32IInstruction *instruction);
	virtual void Visit(Maxwell::ISCADDInstruction *instruction);
	virtual void Visit(Maxwell::ISETPInstruction *instruction);
	virtual void Visit(Maxwell::LOPInstruction *instruction);
	virtual void Visit(Maxwell::LOP32IInstruction *instruction);
	virtual void Visit(Maxwell::POPCInstruction *instruction);
	virtual void Visit(Maxwell::SHLInstruction *instruction);
	virtual void Visit(Maxwell::SHRInstruction *instruction);
	virtual void Visit(Maxwell::XMADInstruction *instruction);

	virtual void Visit(Volta::FLOInstruction *instruction);
	virtual void Visit(Volta::IABSInstruction *instruction);
	virtual void Visit(Volta::IADD3Instruction *instruction);
	virtual void Visit(Volta::IMADInstruction *instruction);
	virtual void Visit(Volta::ISETPInstruction *instruction);
	virtual void Visit(Volta::LOP3Instruction *instruction);
	virtual void Visit(Volta::SHFInstruction *instruction);

	// Float

	virtual void Visit(Maxwell::DADDInstruction *instruction);
	virtual void Visit(Maxwell::DFMAInstruction *instruction);
	virtual void Visit(Maxwell::DMNMXInstruction *instruction);
	virtual void Visit(Maxwell::DMULInstruction *instruction);
	virtual void Visit(Maxwell::DSETPInstruction *instruction);
	virtual void Visit(Maxwell::FADDInstruction *instruction);
	virtual void Visit(Maxwell::MUFUInstruction *instruction);

	virtual void Visit(Volta::DADDInstruction *instruction);
	virtual void Visit(Volta::DFMAInstruction *instruction);
	virtual void Visit(Volta::DMULInstruction *instruction);
	virtual void Visit(Volta::DSETPInstruction *instruction);
	virtual void Visit(Volta::FADDInstruction *instruction);
	virtual void Visit(Volta::MUFUInstruction *instruction);

	// LoadStore

	virtual void Visit(Maxwell::ATOMInstruction *instruction);
	virtual void Visit(Maxwell::ATOMCASInstruction *instruction);
	virtual void Visit(Maxwell::LDGInstruction *instruction);
	virtual void Visit(Maxwell::LDSInstruction *instruction);
	virtual void Visit(Maxwell::MEMBARInstruction *instruction);
	virtual void Visit(Maxwell::REDInstruction *instruction);
	virtual void Visit(Maxwell::STGInstruction *instruction);
	virtual void Visit(Maxwell::STSInstruction *instruction);

	virtual void Visit(Volta::ATOMGInstruction *instruction);
	virtual void Visit(Volta::LDGInstruction *instruction);
	virtual void Visit(Volta::LDSInstruction *instruction);
	virtual void Visit(Volta::MEMBARInstruction *instruction);
	virtual void Visit(Volta::REDInstruction *instruction);
	virtual void Visit(Volta::STGInstruction *instruction);
	virtual void Visit(Volta::STSInstruction *instruction);

	// Misc

	virtual void Visit(Maxwell::CS2RInstruction *instruction);
	virtual void Visit(Maxwell::DEPBARInstruction *instruction);
	virtual void Visit(Maxwell::BARInstruction *instruction);
	virtual void Visit(Maxwell::NOPInstruction *instruction);
	virtual void Visit(Maxwell::S2RInstruction *instruction);

	virtual void Visit(Volta::CS2RInstruction *instruction);
	virtual void Visit(Volta::DEPBARInstruction *instruction);
	virtual void Visit(Volta::BARInstruction *instruction);
	virtual void Visit(Volta::NOPInstruction *instruction);
	virtual void Visit(Volta::S2RInstruction *instruction);

	// Movement

	virtual void Visit(Maxwell::MOVInstruction *instruction);
	virtual void Visit(Maxwell::MOV32IInstruction *instruction);
	virtual void Visit(Maxwell::PRMTInstruction *instruction);
	virtual void Visit(Maxwell::SELInstruction *instruction);
	virtual void Visit(Maxwell::SHFLInstruction *instruction);

	virtual void Visit(Volta::MOVInstruction *instruction);
	virtual void Visit(Volta::PRMTInstruction *instruction);
	virtual void Visit(Volta::SELInstruction *instruction);
	virtual void Visit(Volta::SHFLInstruction *instruction);

	// Predicate

	virtual void Visit(Maxwell::PSETPInstruction *instruction);

	virtual void Visit(Volta::PLOP3Instruction *instruction);

	// Operands

	virtual void Visit(Operand *operand);

	virtual void Visit(Address *address);
	virtual void Visit(Composite *composite);
	virtual void Visit(Constant *constant);

	virtual void Visit(Immediate *immediate);
	virtual void Visit(I8Immediate *immediate);
	virtual void Visit(I16Immediate *immediate);
	virtual void Visit(I32Immediate *immediate);
	virtual void Visit(F32Immediate *immediate);

	virtual void Visit(Register *reg);
	virtual void Visit(Predicate *reg);
	virtual void Visit(SpecialRegister *reg);
	virtual void Visit(CarryFlag *carry);
};

}
