#pragma once

namespace SASS {
	constexpr unsigned int COMPUTE_MIN = 52;
	constexpr unsigned int COMPUTE_MAX = 61;
}

// Utils

#include "SASS/BinaryUtils.h"
#include "SASS/Constants.h"

// Structure

#include "SASS/Program.h"
#include "SASS/Function.h"
#include "SASS/BasicBlock.h"

#include "SASS/GlobalVariable.h"
#include "SASS/Relocation.h"

// Instructions

#include "SASS/Instructions/Instruction.h"
#include "SASS/Instructions/PredicatedInstruction.h"
#include "SASS/Instructions/SCHIInstruction.h"

#include "SASS/Instructions/Control/BRAInstruction.h"
#include "SASS/Instructions/Control/BRKInstruction.h"
#include "SASS/Instructions/Control/CONTInstruction.h"
#include "SASS/Instructions/Control/EXITInstruction.h"
#include "SASS/Instructions/Control/PBKInstruction.h"
#include "SASS/Instructions/Control/PCNTInstruction.h"
#include "SASS/Instructions/Control/RETInstruction.h"
#include "SASS/Instructions/Control/SSYInstruction.h"
#include "SASS/Instructions/Control/SYNCInstruction.h"

#include "SASS/Instructions/Conversion/F2IInstruction.h"
#include "SASS/Instructions/Conversion/I2FInstruction.h"
#include "SASS/Instructions/Conversion/I2IInstruction.h"

#include "SASS/Instructions/Integer/BFEInstruction.h"
#include "SASS/Instructions/Integer/IADDInstruction.h"
#include "SASS/Instructions/Integer/IADD3Instruction.h"
#include "SASS/Instructions/Integer/IADD32IInstruction.h"
#include "SASS/Instructions/Integer/ISCADDInstruction.h"
#include "SASS/Instructions/Integer/ISETPInstruction.h"
#include "SASS/Instructions/Integer/LOPInstruction.h"
#include "SASS/Instructions/Integer/LOP32IInstruction.h"
#include "SASS/Instructions/Integer/POPCInstruction.h"
#include "SASS/Instructions/Integer/SHLInstruction.h"
#include "SASS/Instructions/Integer/SHRInstruction.h"
#include "SASS/Instructions/Integer/XMADInstruction.h"

#include "SASS/Instructions/Float/DADDInstruction.h"
#include "SASS/Instructions/Float/DMNMXInstruction.h"
#include "SASS/Instructions/Float/DMULInstruction.h"
#include "SASS/Instructions/Float/DSETPInstruction.h"
#include "SASS/Instructions/Float/MUFUInstruction.h"

#include "SASS/Instructions/LoadStore/LDGInstruction.h"
#include "SASS/Instructions/LoadStore/MEMBARInstruction.h"
#include "SASS/Instructions/LoadStore/REDInstruction.h"
#include "SASS/Instructions/LoadStore/STGInstruction.h"

#include "SASS/Instructions/Misc/CS2RInstruction.h"
#include "SASS/Instructions/Misc/DEPBARInstruction.h"
#include "SASS/Instructions/Misc/BARInstruction.h"
#include "SASS/Instructions/Misc/NOPInstruction.h"
#include "SASS/Instructions/Misc/S2RInstruction.h"

#include "SASS/Instructions/Movement/MOVInstruction.h"
#include "SASS/Instructions/Movement/MOV32IInstruction.h"
#include "SASS/Instructions/Movement/SELInstruction.h"
#include "SASS/Instructions/Movement/SHFLInstruction.h"

#include "SASS/Instructions/Predicate/PSETPInstruction.h"

// Operands

#include "SASS/Operands/Operand.h"

#include "SASS/Operands/Address.h"
#include "SASS/Operands/Constant.h"

#include "SASS/Operands/Composite.h"

#include "SASS/Operands/Immediate.h"
#include "SASS/Operands/I8Immediate.h"
#include "SASS/Operands/I16Immediate.h"
#include "SASS/Operands/I32Immediate.h"
#include "SASS/Operands/F32Immediate.h"

#include "SASS/Operands/Register.h"
#include "SASS/Operands/SpecialRegister.h"
