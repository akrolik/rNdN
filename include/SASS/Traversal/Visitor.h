#pragma once

namespace SASS {

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

class Visitor
{
public:
	//TODO: Add in remaining visitors

	// Instructions

	virtual void Visit(Instruction *instruction) {}
	virtual void Visit(PredicatedInstruction *instruction) {}
	virtual void Visit(SCHIInstruction *instruction) {}

	// Control

	virtual void Visit(DivergenceInstruction *instruction) {}

	virtual void Visit(SSYInstruction *instruction);
	virtual void Visit(PBKInstruction *instruction);
	virtual void Visit(PCNTInstruction *instruction);

	virtual void Visit(SYNCInstruction *instruction) {}
	virtual void Visit(BRKInstruction *instruction) {}
	virtual void Visit(CONTInstruction *instruction) {}

	virtual void Visit(BRAInstruction *instruction) {}
	virtual void Visit(EXITInstruction *instruction) {}
	virtual void Visit(RETInstruction *instruction) {}

	// Conversion

	virtual void Visit(F2IInstruction *instruction) {}
	virtual void Visit(I2FInstruction *instruction) {}
	virtual void Visit(I2IInstruction *instruction) {}

	// Integer

	virtual void Visit(BFEInstruction *instruction) {}
	virtual void Visit(FLOInstruction *instruction) {}
	virtual void Visit(IADDInstruction *instruction) {}
	virtual void Visit(IADD3Instruction *instruction) {}
	virtual void Visit(IADD32IInstruction *instruction) {}
	virtual void Visit(ISCADDInstruction *instruction) {}
	virtual void Visit(ISETPInstruction *instruction) {}
	virtual void Visit(LOPInstruction *instruction) {}
	virtual void Visit(LOP32IInstruction *instruction) {}
	virtual void Visit(POPCInstruction *instruction) {}
	virtual void Visit(SHLInstruction *instruction) {}
	virtual void Visit(SHRInstruction *instruction) {}
	virtual void Visit(XMADInstruction *instruction) {}

	// Float

	virtual void Visit(DADDInstruction *instruction) {}
	virtual void Visit(DFMAInstruction *instruction) {}
	virtual void Visit(DMNMXInstruction *instruction) {}
	virtual void Visit(DMULInstruction *instruction) {}
	virtual void Visit(DSETPInstruction *instruction) {}
	virtual void Visit(MUFUInstruction *instruction) {}

	// LoadStore

	virtual void Visit(ATOMInstruction *instruction) {}
	virtual void Visit(LDGInstruction *instruction) {}
	virtual void Visit(LDSInstruction *instruction) {}
	virtual void Visit(MEMBARInstruction *instruction) {}
	virtual void Visit(REDInstruction *instruction) {}
	virtual void Visit(STGInstruction *instruction) {}
	virtual void Visit(STSInstruction *instruction) {}

	// Misc

	virtual void Visit(CS2RInstruction *instruction) {}
	virtual void Visit(DEPBARInstruction *instruction) {}
	virtual void Visit(BARInstruction *instruction) {}
	virtual void Visit(NOPInstruction *instruction) {}
	virtual void Visit(S2RInstruction *instruction) {}

	// Movement

	virtual void Visit(MOVInstruction *instruction) {}
	virtual void Visit(MOV32IInstruction *instruction) {}
	virtual void Visit(SELInstruction *instruction) {}
	virtual void Visit(SHFLInstruction *instruction) {}

	// Predicate

	virtual void Visit(PSETPInstruction *instruction) {}
};

}
