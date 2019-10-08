#pragma once

// Utility

#include "PTX/Concepts.h"
#include "PTX/StateSpace.h"
#include "PTX/Tuple.h"
#include "PTX/Type.h"
#include "PTX/Utils.h"

// Structure

#include "PTX/Program.h"
#include "PTX/Module.h"

// Declarations

#include "PTX/Declarations/Declaration.h"
#include "PTX/Declarations/SpecialRegisterDeclarations.h"
#include "PTX/Declarations/VariableDeclaration.h"

// Functions

#include "PTX/FunctionOptions.h"
#include "PTX/Functions/Function.h"
#include "PTX/Functions/FunctionDeclaration.h"
#include "PTX/Functions/FunctionDeclarationBase.h"
#include "PTX/Functions/FunctionDefinition.h"
#include "PTX/Functions/ExternalMathFunctions.h"

// Statements

#include "PTX/Statements/Statement.h"
#include "PTX/Statements/Label.h"

#include "PTX/Statements/BlankStatement.h"
#include "PTX/Statements/CommentStatement.h"
#include "PTX/Statements/DirectiveStatement.h"
#include "PTX/Statements/InstructionStatement.h"

#include "PTX/Statements/StatementList.h"
#include "PTX/Statements/BlockStatement.h"

// Operands

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/BracedOperand.h"
#include "PTX/Operands/Constant.h"
#include "PTX/Operands/SpecialConstants.h"
#include "PTX/Operands/Value.h"

// Extended operand formats

#include "PTX/Operands/Extended/DualOperand.h"
#include "PTX/Operands/Extended/HexOperand.h"
#include "PTX/Operands/Extended/InvertedOperand.h"
#include "PTX/Operands/Extended/ListOperand.h"
#include "PTX/Operands/Extended/StringOperand.h"

// Variables

#include "PTX/Operands/Variables/Variable.h"
#include "PTX/Operands/Variables/AddressableVariable.h"
#include "PTX/Operands/Variables/Register.h"
#include "PTX/Operands/Variables/BracedRegister.h"
#include "PTX/Operands/Variables/SinkRegister.h"
#include "PTX/Operands/Variables/IndexedRegister.h"

// Addresses

#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Address/DereferencedAddress.h"
#include "PTX/Operands/Address/MemoryAddress.h"
#include "PTX/Operands/Address/RegisterAddress.h"

// Type/space adapters

#include "PTX/Operands/Adapters/Adapter.h"
#include "PTX/Operands/Adapters/BitAdapter.h"
#include "PTX/Operands/Adapters/ArrayAdapter.h"
#include "PTX/Operands/Adapters/PointerAdapter.h"
#include "PTX/Operands/Adapters/SignedAdapter.h"
#include "PTX/Operands/Adapters/TruncateAdapter.h"
#include "PTX/Operands/Adapters/TypeAdapter.h"
#include "PTX/Operands/Adapters/UnsignedAdapter.h"
#include "PTX/Operands/Adapters/VariableAdapter.h"

// Instructions

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/DevInstruction.h"

#include "PTX/Instructions/Modifiers/CarryModifier.h"
#include "PTX/Instructions/Modifiers/ComparisonModifier.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/HalfModifier.h"
#include "PTX/Instructions/Modifiers/PredicateModifier.h"
#include "PTX/Instructions/Modifiers/RoundingModifier.h"
#include "PTX/Instructions/Modifiers/SaturateModifier.h"
#include "PTX/Instructions/Modifiers/ScopeModifier.h"
#include "PTX/Instructions/Modifiers/UniformModifier.h"

#include "PTX/Instructions/Arithmetic/AbsoluteInstruction.h"
#include "PTX/Instructions/Arithmetic/AddInstruction.h"
#include "PTX/Instructions/Arithmetic/BitFieldExtractInstruction.h"
#include "PTX/Instructions/Arithmetic/BitFieldInsertInstruction.h"
#include "PTX/Instructions/Arithmetic/BitFindInstruction.h"
#include "PTX/Instructions/Arithmetic/BitReverseInstruction.h"
#include "PTX/Instructions/Arithmetic/CopySignInstruction.h"
#include "PTX/Instructions/Arithmetic/CosineInstruction.h"
#include "PTX/Instructions/Arithmetic/CountLeadingZerosInstruction.h"
#include "PTX/Instructions/Arithmetic/DivideInstruction.h"
#include "PTX/Instructions/Arithmetic/Exp2Instruction.h"
#include "PTX/Instructions/Arithmetic/FindNthBitInstruction.h"
#include "PTX/Instructions/Arithmetic/FMAInstruction.h"
#include "PTX/Instructions/Arithmetic/Log2Instruction.h"
#include "PTX/Instructions/Arithmetic/MADInstruction.h"
#include "PTX/Instructions/Arithmetic/MADWideInstruction.h"
#include "PTX/Instructions/Arithmetic/MaximumInstruction.h"
#include "PTX/Instructions/Arithmetic/MultiplyInstruction.h"
#include "PTX/Instructions/Arithmetic/MultiplyWideInstruction.h"
#include "PTX/Instructions/Arithmetic/MinimumInstruction.h"
#include "PTX/Instructions/Arithmetic/NegateInstruction.h"
#include "PTX/Instructions/Arithmetic/PopulationCountInstruction.h"
#include "PTX/Instructions/Arithmetic/ReciprocalInstruction.h"
#include "PTX/Instructions/Arithmetic/ReciprocalRootInstruction.h"
#include "PTX/Instructions/Arithmetic/RemainderInstruction.h"
#include "PTX/Instructions/Arithmetic/RootInstruction.h"
#include "PTX/Instructions/Arithmetic/SADInstruction.h"
#include "PTX/Instructions/Arithmetic/SineInstruction.h"
#include "PTX/Instructions/Arithmetic/SubtractInstruction.h"
#include "PTX/Instructions/Arithmetic/TestPropertyInstruction.h"

#include "PTX/Instructions/Comparison/SelectInstruction.h"
#include "PTX/Instructions/Comparison/SetInstruction.h"
#include "PTX/Instructions/Comparison/SetPredicateInstruction.h"
#include "PTX/Instructions/Comparison/SignSelectInstruction.h"

#include "PTX/Instructions/ControlFlow/BranchInstruction.h"
#include "PTX/Instructions/ControlFlow/BranchIndexInstruction.h"
#include "PTX/Instructions/ControlFlow/CallInstructionBase.h"
#include "PTX/Instructions/ControlFlow/CallInstruction.h"
#include "PTX/Instructions/ControlFlow/ExitInstruction.h"
#include "PTX/Instructions/ControlFlow/ReturnInstruction.h"

#include "PTX/Instructions/Logical/AndInstruction.h"
#include "PTX/Instructions/Logical/CNotInstruction.h"
#include "PTX/Instructions/Logical/Logical3OpInstruction.h"
#include "PTX/Instructions/Logical/NotInstruction.h"
#include "PTX/Instructions/Logical/OrInstruction.h"
#include "PTX/Instructions/Logical/XorInstruction.h"

#include "PTX/Instructions/Synchronization/FenceInstruction.h"
#include "PTX/Instructions/Synchronization/MemoryBarrierInstruction.h"
#include "PTX/Instructions/Synchronization/ReductionInstruction.h"
#include "PTX/Instructions/Synchronization/VoteInstruction.h"
#include "PTX/Instructions/Synchronization/ActiveMaskInstruction.h"
#include "PTX/Instructions/Synchronization/BarrierWarpInstruction.h"
#include "PTX/Instructions/Synchronization/BarrierInstruction.h"
#include "PTX/Instructions/Synchronization/BarrierReductionInstruction.h"
#include "PTX/Instructions/Synchronization/MatchInstruction.h"
#include "PTX/Instructions/Synchronization/AtomicInstruction.h"

#include "PTX/Instructions/Data/Modifiers/ConvertFlushSubnormalModifier.h"
#include "PTX/Instructions/Data/Modifiers/ConvertRoundingModifier.h"
#include "PTX/Instructions/Data/Modifiers/ConvertSaturateModifier.h"

#include "PTX/Instructions/Data/ConvertAddressInstruction.h"
#include "PTX/Instructions/Data/ConvertInstruction.h"
#include "PTX/Instructions/Data/IsSpaceInstruction.h"
#include "PTX/Instructions/Data/LoadInstruction.h"
#include "PTX/Instructions/Data/LoadNCInstruction.h"
#include "PTX/Instructions/Data/LoadUniformInstruction.h"
#include "PTX/Instructions/Data/MoveAddressInstruction.h"
#include "PTX/Instructions/Data/MoveInstruction.h"
#include "PTX/Instructions/Data/PackInstruction.h"
#include "PTX/Instructions/Data/PermuteInstruction.h"
#include "PTX/Instructions/Data/PrefetchInstruction.h"
#include "PTX/Instructions/Data/PrefetchUniformInstruction.h"
#include "PTX/Instructions/Data/ShuffleInstruction.h"
#include "PTX/Instructions/Data/StoreInstruction.h"
#include "PTX/Instructions/Data/UnpackInstruction.h"

#include "PTX/Instructions/Shift/ShiftLeftInstruction.h"
#include "PTX/Instructions/Shift/ShiftRightInstruction.h"
#include "PTX/Instructions/Shift/FunnelShiftInstruction.h"
