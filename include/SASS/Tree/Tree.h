#pragma once

namespace SASS {
	constexpr unsigned int COMPUTE_MIN = 52;
	constexpr unsigned int COMPUTE_MAX = 61;
}

// Utils

#include "SASS/Tree/BinaryUtils.h"
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

#include "SASS/Tree/Instructions/Instruction.h"
#include "SASS/Tree/Instructions/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/SCHIInstruction.h"

#include "SASS/Tree/Instructions/Control/BRAInstruction.h"
#include "SASS/Tree/Instructions/Control/BRKInstruction.h"
#include "SASS/Tree/Instructions/Control/CONTInstruction.h"
#include "SASS/Tree/Instructions/Control/EXITInstruction.h"
#include "SASS/Tree/Instructions/Control/PBKInstruction.h"
#include "SASS/Tree/Instructions/Control/PCNTInstruction.h"
#include "SASS/Tree/Instructions/Control/RETInstruction.h"
#include "SASS/Tree/Instructions/Control/SSYInstruction.h"
#include "SASS/Tree/Instructions/Control/SYNCInstruction.h"

#include "SASS/Tree/Instructions/Conversion/F2IInstruction.h"
#include "SASS/Tree/Instructions/Conversion/I2FInstruction.h"
#include "SASS/Tree/Instructions/Conversion/I2IInstruction.h"

#include "SASS/Tree/Instructions/Integer/BFEInstruction.h"
#include "SASS/Tree/Instructions/Integer/FLOInstruction.h"
#include "SASS/Tree/Instructions/Integer/IADDInstruction.h"
#include "SASS/Tree/Instructions/Integer/IADD3Instruction.h"
#include "SASS/Tree/Instructions/Integer/IADD32IInstruction.h"
#include "SASS/Tree/Instructions/Integer/ISCADDInstruction.h"
#include "SASS/Tree/Instructions/Integer/ISETPInstruction.h"
#include "SASS/Tree/Instructions/Integer/LOPInstruction.h"
#include "SASS/Tree/Instructions/Integer/LOP32IInstruction.h"
#include "SASS/Tree/Instructions/Integer/POPCInstruction.h"
#include "SASS/Tree/Instructions/Integer/SHLInstruction.h"
#include "SASS/Tree/Instructions/Integer/SHRInstruction.h"
#include "SASS/Tree/Instructions/Integer/XMADInstruction.h"

#include "SASS/Tree/Instructions/Float/DADDInstruction.h"
#include "SASS/Tree/Instructions/Float/DFMAInstruction.h"
#include "SASS/Tree/Instructions/Float/DMNMXInstruction.h"
#include "SASS/Tree/Instructions/Float/DMULInstruction.h"
#include "SASS/Tree/Instructions/Float/DSETPInstruction.h"
#include "SASS/Tree/Instructions/Float/MUFUInstruction.h"

#include "SASS/Tree/Instructions/LoadStore/ATOMInstruction.h"
#include "SASS/Tree/Instructions/LoadStore/ATOMCASInstruction.h"
#include "SASS/Tree/Instructions/LoadStore/LDGInstruction.h"
#include "SASS/Tree/Instructions/LoadStore/LDSInstruction.h"
#include "SASS/Tree/Instructions/LoadStore/MEMBARInstruction.h"
#include "SASS/Tree/Instructions/LoadStore/REDInstruction.h"
#include "SASS/Tree/Instructions/LoadStore/STGInstruction.h"
#include "SASS/Tree/Instructions/LoadStore/STSInstruction.h"

#include "SASS/Tree/Instructions/Misc/CS2RInstruction.h"
#include "SASS/Tree/Instructions/Misc/DEPBARInstruction.h"
#include "SASS/Tree/Instructions/Misc/BARInstruction.h"
#include "SASS/Tree/Instructions/Misc/NOPInstruction.h"
#include "SASS/Tree/Instructions/Misc/S2RInstruction.h"

#include "SASS/Tree/Instructions/Movement/MOVInstruction.h"
#include "SASS/Tree/Instructions/Movement/MOV32IInstruction.h"
#include "SASS/Tree/Instructions/Movement/SELInstruction.h"
#include "SASS/Tree/Instructions/Movement/SHFLInstruction.h"

#include "SASS/Tree/Instructions/Predicate/PSETPInstruction.h"

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

#include "SASS/Tree/Operands/Register.h"
#include "SASS/Tree/Operands/Predicate.h"
#include "SASS/Tree/Operands/SpecialRegister.h"
#include "SASS/Tree/Operands/CarryFlag.h"
