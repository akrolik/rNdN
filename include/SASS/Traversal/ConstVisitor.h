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
	class SELInstruction;
	class SHFLInstruction;
}

namespace Volta {
	class MOVInstruction;
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

	virtual void Visit(const Maxwell::Instruction *instruction);
	virtual void Visit(const Maxwell::PredicatedInstruction *instruction);
	virtual void Visit(const Maxwell::SCHIInstruction *instruction);

	virtual void Visit(const Volta::Instruction *instruction);
	virtual void Visit(const Volta::PredicatedInstruction *instruction);

	// Control

	virtual void Visit(const Maxwell::DivergenceInstruction *instruction);

	virtual void Visit(const Maxwell::SSYInstruction *instruction);
	virtual void Visit(const Maxwell::PBKInstruction *instruction);
	virtual void Visit(const Maxwell::PCNTInstruction *instruction);

	virtual void Visit(const Maxwell::SYNCInstruction *instruction);
	virtual void Visit(const Maxwell::BRKInstruction *instruction);
	virtual void Visit(const Maxwell::CONTInstruction *instruction);

	virtual void Visit(const Maxwell::BRAInstruction *instruction);
	virtual void Visit(const Maxwell::EXITInstruction *instruction);
	virtual void Visit(const Maxwell::RETInstruction *instruction);

	virtual void Visit(const Volta::ControlInstruction *instruction);
	virtual void Visit(const Volta::DivergenceInstruction *instruction);

	virtual void Visit(const Volta::BRAInstruction *instruction);
	virtual void Visit(const Volta::EXITInstruction *instruction);
	virtual void Visit(const Volta::BSSYInstruction *instruction);
	virtual void Visit(const Volta::BSYNCInstruction *instruction);

	// Conversion

	virtual void Visit(const Maxwell::F2FInstruction *instruction);
	virtual void Visit(const Maxwell::F2IInstruction *instruction);
	virtual void Visit(const Maxwell::I2FInstruction *instruction);
	virtual void Visit(const Maxwell::I2IInstruction *instruction);

	virtual void Visit(const Volta::F2FInstruction *instruction);
	virtual void Visit(const Volta::F2IInstruction *instruction);
	virtual void Visit(const Volta::I2FInstruction *instruction);
	virtual void Visit(const Volta::I2IInstruction *instruction);
	virtual void Visit(const Volta::FRNDInstruction *instruction);

	// Integer

	virtual void Visit(const Maxwell::BFEInstruction *instruction);
	virtual void Visit(const Maxwell::FLOInstruction *instruction);
	virtual void Visit(const Maxwell::IADDInstruction *instruction);
	virtual void Visit(const Maxwell::IADD3Instruction *instruction);
	virtual void Visit(const Maxwell::IADD32IInstruction *instruction);
	virtual void Visit(const Maxwell::ISCADDInstruction *instruction);
	virtual void Visit(const Maxwell::ISETPInstruction *instruction);
	virtual void Visit(const Maxwell::LOPInstruction *instruction);
	virtual void Visit(const Maxwell::LOP32IInstruction *instruction);
	virtual void Visit(const Maxwell::POPCInstruction *instruction);
	virtual void Visit(const Maxwell::SHLInstruction *instruction);
	virtual void Visit(const Maxwell::SHRInstruction *instruction);
	virtual void Visit(const Maxwell::XMADInstruction *instruction);

	virtual void Visit(const Volta::FLOInstruction *instruction);
	virtual void Visit(const Volta::IADD3Instruction *instruction);
	virtual void Visit(const Volta::IMADInstruction *instruction);
	virtual void Visit(const Volta::ISETPInstruction *instruction);
	virtual void Visit(const Volta::LOP3Instruction *instruction);
	virtual void Visit(const Volta::SHFInstruction *instruction);

	// Float

	virtual void Visit(const Maxwell::DADDInstruction *instruction);
	virtual void Visit(const Maxwell::DFMAInstruction *instruction);
	virtual void Visit(const Maxwell::DMNMXInstruction *instruction);
	virtual void Visit(const Maxwell::DMULInstruction *instruction);
	virtual void Visit(const Maxwell::DSETPInstruction *instruction);
	virtual void Visit(const Maxwell::FADDInstruction *instruction);
	virtual void Visit(const Maxwell::MUFUInstruction *instruction);

	virtual void Visit(const Volta::DADDInstruction *instruction);
	virtual void Visit(const Volta::DFMAInstruction *instruction);
	virtual void Visit(const Volta::DMULInstruction *instruction);
	virtual void Visit(const Volta::DSETPInstruction *instruction);
	virtual void Visit(const Volta::FADDInstruction *instruction);
	virtual void Visit(const Volta::MUFUInstruction *instruction);

	// LoadStore

	virtual void Visit(const Maxwell::ATOMInstruction *instruction);
	virtual void Visit(const Maxwell::ATOMCASInstruction *instruction);
	virtual void Visit(const Maxwell::LDGInstruction *instruction);
	virtual void Visit(const Maxwell::LDSInstruction *instruction);
	virtual void Visit(const Maxwell::MEMBARInstruction *instruction);
	virtual void Visit(const Maxwell::REDInstruction *instruction);
	virtual void Visit(const Maxwell::STGInstruction *instruction);
	virtual void Visit(const Maxwell::STSInstruction *instruction);

	virtual void Visit(const Volta::ATOMGInstruction *instruction);
	virtual void Visit(const Volta::LDGInstruction *instruction);
	virtual void Visit(const Volta::LDSInstruction *instruction);
	virtual void Visit(const Volta::MEMBARInstruction *instruction);
	virtual void Visit(const Volta::REDInstruction *instruction);
	virtual void Visit(const Volta::STGInstruction *instruction);
	virtual void Visit(const Volta::STSInstruction *instruction);

	// Misc

	virtual void Visit(const Maxwell::CS2RInstruction *instruction);
	virtual void Visit(const Maxwell::DEPBARInstruction *instruction);
	virtual void Visit(const Maxwell::BARInstruction *instruction);
	virtual void Visit(const Maxwell::NOPInstruction *instruction);
	virtual void Visit(const Maxwell::S2RInstruction *instruction);

	virtual void Visit(const Volta::CS2RInstruction *instruction);
	virtual void Visit(const Volta::DEPBARInstruction *instruction);
	virtual void Visit(const Volta::BARInstruction *instruction);
	virtual void Visit(const Volta::NOPInstruction *instruction);
	virtual void Visit(const Volta::S2RInstruction *instruction);

	// Movement

	virtual void Visit(const Maxwell::MOVInstruction *instruction);
	virtual void Visit(const Maxwell::MOV32IInstruction *instruction);
	virtual void Visit(const Maxwell::SELInstruction *instruction);
	virtual void Visit(const Maxwell::SHFLInstruction *instruction);

	virtual void Visit(const Volta::MOVInstruction *instruction);
	virtual void Visit(const Volta::SELInstruction *instruction);
	virtual void Visit(const Volta::SHFLInstruction *instruction);

	// Predicate

	virtual void Visit(const Maxwell::PSETPInstruction *instruction);

	virtual void Visit(const Volta::PLOP3Instruction *instruction);

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
