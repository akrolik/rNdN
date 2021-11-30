#pragma once

// Utils

#include "SASS/Tree/Constants.h"

// Structure

#include "SASS/Tree/Node.h"
#include "SASS/Tree/Program.h"
#include "SASS/Tree/Function.h"
#include "SASS/Tree/BasicBlock.h"

#include "SASS/Tree/Variable.h"
#include "SASS/Tree/GlobalVariable.h"
#include "SASS/Tree/SharedVariable.h"
#include "SASS/Tree/DynamicSharedVariable.h"

#include "SASS/Tree/Relocation.h"
#include "SASS/Tree/IndirectBranch.h"

// Instructions

#include "SASS/Tree/Instructions/BinaryUtils.h"
#include "SASS/Tree/Instructions/Instruction.h"

// Maxwell/Pascal instructions

#include "SASS/Tree/Instructions/Maxwell/BinaryUtils.h"
#include "SASS/Tree/Instructions/Maxwell/Instruction.h"
#include "SASS/Tree/Instructions/Maxwell/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/SCHIInstruction.h"

#include "SASS/Tree/Instructions/Maxwell/Control/DivergenceInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Control/BRAInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Control/BRKInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Control/CONTInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Control/EXITInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Control/PBKInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Control/PCNTInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Control/RETInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Control/SSYInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Control/SYNCInstruction.h"

#include "SASS/Tree/Instructions/Maxwell/Conversion/F2IInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Conversion/F2FInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Conversion/I2FInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Conversion/I2IInstruction.h"

#include "SASS/Tree/Instructions/Maxwell/Integer/BFEInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Integer/FLOInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Integer/IADDInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Integer/IADD3Instruction.h"
#include "SASS/Tree/Instructions/Maxwell/Integer/ICMPInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Integer/IADD32IInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Integer/ISCADDInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Integer/ISETPInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Integer/LOPInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Integer/LOP32IInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Integer/POPCInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Integer/SHLInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Integer/SHRInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Integer/XMADInstruction.h"

#include "SASS/Tree/Instructions/Maxwell/Float/DADDInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Float/DFMAInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Float/DMNMXInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Float/DMULInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Float/DSETPInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Float/FADDInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Float/MUFUInstruction.h"

#include "SASS/Tree/Instructions/Maxwell/LoadStore/ATOMInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/LoadStore/ATOMCASInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/LoadStore/LDGInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/LoadStore/LDSInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/LoadStore/MEMBARInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/LoadStore/REDInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/LoadStore/STGInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/LoadStore/STSInstruction.h"

#include "SASS/Tree/Instructions/Maxwell/Misc/CS2RInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Misc/DEPBARInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Misc/BARInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Misc/NOPInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Misc/S2RInstruction.h"

#include "SASS/Tree/Instructions/Maxwell/Movement/MOVInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Movement/MOV32IInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Movement/PRMTInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Movement/SELInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/Movement/SHFLInstruction.h"

#include "SASS/Tree/Instructions/Maxwell/Predicate/PSETPInstruction.h"

// Volta/Turing/Ampere instructions

#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"
#include "SASS/Tree/Instructions/Volta/Instruction.h"
#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"

#include "SASS/Tree/Instructions/Volta/Control/ControlInstruction.h"
#include "SASS/Tree/Instructions/Volta/Control/DivergenceInstruction.h"
#include "SASS/Tree/Instructions/Volta/Control/BRAInstruction.h"
#include "SASS/Tree/Instructions/Volta/Control/EXITInstruction.h"
#include "SASS/Tree/Instructions/Volta/Control/BSSYInstruction.h"
#include "SASS/Tree/Instructions/Volta/Control/BSYNCInstruction.h"

#include "SASS/Tree/Instructions/Volta/Conversion/F2IInstruction.h"
#include "SASS/Tree/Instructions/Volta/Conversion/F2FInstruction.h"
#include "SASS/Tree/Instructions/Volta/Conversion/I2FInstruction.h"
#include "SASS/Tree/Instructions/Volta/Conversion/I2IInstruction.h"
#include "SASS/Tree/Instructions/Volta/Conversion/FRNDInstruction.h"

#include "SASS/Tree/Instructions/Volta/Float/DADDInstruction.h"
#include "SASS/Tree/Instructions/Volta/Float/DFMAInstruction.h"
#include "SASS/Tree/Instructions/Volta/Float/DMULInstruction.h"
#include "SASS/Tree/Instructions/Volta/Float/DSETPInstruction.h"
#include "SASS/Tree/Instructions/Volta/Float/FADDInstruction.h"
#include "SASS/Tree/Instructions/Volta/Float/MUFUInstruction.h"

#include "SASS/Tree/Instructions/Volta/Integer/FLOInstruction.h"
#include "SASS/Tree/Instructions/Volta/Integer/IABSInstruction.h"
#include "SASS/Tree/Instructions/Volta/Integer/IADD3Instruction.h"
#include "SASS/Tree/Instructions/Volta/Integer/IMADInstruction.h"
#include "SASS/Tree/Instructions/Volta/Integer/ISETPInstruction.h"
#include "SASS/Tree/Instructions/Volta/Integer/LOP3Instruction.h"
#include "SASS/Tree/Instructions/Volta/Integer/SHFInstruction.h"

#include "SASS/Tree/Instructions/Volta/LoadStore/ATOMGInstruction.h"
#include "SASS/Tree/Instructions/Volta/LoadStore/LDGInstruction.h"
#include "SASS/Tree/Instructions/Volta/LoadStore/LDSInstruction.h"
#include "SASS/Tree/Instructions/Volta/LoadStore/MEMBARInstruction.h"
#include "SASS/Tree/Instructions/Volta/LoadStore/STGInstruction.h"
#include "SASS/Tree/Instructions/Volta/LoadStore/STSInstruction.h"
#include "SASS/Tree/Instructions/Volta/LoadStore/REDInstruction.h"

#include "SASS/Tree/Instructions/Volta/Misc/NOPInstruction.h"
#include "SASS/Tree/Instructions/Volta/Misc/BARInstruction.h"
#include "SASS/Tree/Instructions/Volta/Misc/DEPBARInstruction.h"
#include "SASS/Tree/Instructions/Volta/Misc/S2RInstruction.h"
#include "SASS/Tree/Instructions/Volta/Misc/CS2RInstruction.h"

#include "SASS/Tree/Instructions/Volta/Movement/MOVInstruction.h"
#include "SASS/Tree/Instructions/Volta/Movement/PRMTInstruction.h"
#include "SASS/Tree/Instructions/Volta/Movement/SELInstruction.h"
#include "SASS/Tree/Instructions/Volta/Movement/SHFLInstruction.h"

#include "SASS/Tree/Instructions/Volta/Predicate/PLOP3Instruction.h"

// Operands

#include "SASS/Tree/Operands/Operand.h"

#include "SASS/Tree/Operands/Address.h"
#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Constant.h"

#include "SASS/Tree/Operands/Immediate.h"
#include "SASS/Tree/Operands/I8Immediate.h"
#include "SASS/Tree/Operands/I16Immediate.h"
#include "SASS/Tree/Operands/I32Immediate.h"
#include "SASS/Tree/Operands/F32Immediate.h"
#include "SASS/Tree/Operands/F64Immediate.h"

#include "SASS/Tree/Operands/Register.h"
#include "SASS/Tree/Operands/Predicate.h"
#include "SASS/Tree/Operands/SpecialRegister.h"
#include "SASS/Tree/Operands/CarryFlag.h"
