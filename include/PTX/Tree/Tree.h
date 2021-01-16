#pragma once

namespace PTX {
	constexpr unsigned int MAJOR_VERSION = 6;
	constexpr unsigned int MINOR_VERSION = 3;
}

// Utility

#include "PTX/Tree/Concepts.h"
#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Tuple.h"
#include "PTX/Tree/Type.h"
#include "PTX/Tree/Utils.h"

// Structure

#include "PTX/Tree/Node.h"

#include "PTX/Tree/Program.h"
#include "PTX/Tree/Module.h"
#include "PTX/Tree/BasicBlock.h"

// Directives

#include "PTX/Tree/Directives/Directive.h"
#include "PTX/Tree/Directives/FileDirective.h"
#include "PTX/Tree/Directives/LocationDirective.h"

// Declarations

#include "PTX/Tree/Declarations/Declaration.h"
#include "PTX/Tree/Declarations/SpecialRegisterDeclarations.h"
#include "PTX/Tree/Declarations/VariableDeclaration.h"

// Functions

#include "PTX/Tree/FunctionOptions.h"
#include "PTX/Tree/Functions/Function.h"
#include "PTX/Tree/Functions/FunctionDeclaration.h"
#include "PTX/Tree/Functions/FunctionDeclarationBase.h"
#include "PTX/Tree/Functions/FunctionDefinition.h"
#include "PTX/Tree/Functions/ExternalMathFunctions.h"

// Statements

#include "PTX/Tree/Statements/Statement.h"
#include "PTX/Tree/Statements/LabelStatement.h"

#include "PTX/Tree/Statements/CommentStatement.h"
#include "PTX/Tree/Statements/DeclarationStatement.h"
#include "PTX/Tree/Statements/DirectiveStatement.h"
#include "PTX/Tree/Statements/InstructionStatement.h"

#include "PTX/Tree/Statements/StatementList.h"
#include "PTX/Tree/Statements/BlockStatement.h"

// Operands

#include "PTX/Tree/Operands/Operand.h"
#include "PTX/Tree/Operands/Label.h"
#include "PTX/Tree/Operands/BracedOperand.h"
#include "PTX/Tree/Operands/Constant.h"
#include "PTX/Tree/Operands/SpecialConstants.h"
#include "PTX/Tree/Operands/Value.h"

// Extended operand formats

#include "PTX/Tree/Operands/Extended/DualOperand.h"
#include "PTX/Tree/Operands/Extended/HexOperand.h"
#include "PTX/Tree/Operands/Extended/InvertedOperand.h"
#include "PTX/Tree/Operands/Extended/ListOperand.h"
#include "PTX/Tree/Operands/Extended/StringOperand.h"

// Variables

#include "PTX/Tree/Operands/Variables/Variable.h"

#include "PTX/Tree/Operands/Variables/ConstVariable.h"
#include "PTX/Tree/Operands/Variables/GlobalVariable.h"
#include "PTX/Tree/Operands/Variables/LocalVariable.h"
#include "PTX/Tree/Operands/Variables/ParameterVariable.h"
#include "PTX/Tree/Operands/Variables/SharedVariable.h"

#include "PTX/Tree/Operands/Variables/Register.h"
#include "PTX/Tree/Operands/Variables/BracedRegister.h"
#include "PTX/Tree/Operands/Variables/SinkRegister.h"
#include "PTX/Tree/Operands/Variables/IndexedRegister.h"

// Addresses

#include "PTX/Tree/Operands/Address/Address.h"
#include "PTX/Tree/Operands/Address/DereferencedAddress.h"
#include "PTX/Tree/Operands/Address/MemoryAddress.h"
#include "PTX/Tree/Operands/Address/RegisterAddress.h"

// Type/space adapters

#include "PTX/Tree/Operands/Adapters/Adapter.h"
#include "PTX/Tree/Operands/Adapters/AddressAdapter.h"
#include "PTX/Tree/Operands/Adapters/ArrayAdapter.h"
#include "PTX/Tree/Operands/Adapters/BitAdapter.h"
#include "PTX/Tree/Operands/Adapters/ExtendAdapter.h"
#include "PTX/Tree/Operands/Adapters/PointerAdapter.h"
#include "PTX/Tree/Operands/Adapters/SignedAdapter.h"
#include "PTX/Tree/Operands/Adapters/TruncateAdapter.h"
#include "PTX/Tree/Operands/Adapters/TypeAdapter.h"
#include "PTX/Tree/Operands/Adapters/UnsignedAdapter.h"
#include "PTX/Tree/Operands/Adapters/VariableAdapter.h"

// Instructions

#include "PTX/Tree/Instructions/PredicatedInstruction.h"
#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/DevInstruction.h"

#include "PTX/Tree/Instructions/Modifiers/CarryModifier.h"
#include "PTX/Tree/Instructions/Modifiers/ComparisonModifier.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Tree/Instructions/Modifiers/HalfModifier.h"
#include "PTX/Tree/Instructions/Modifiers/PredicateModifier.h"
#include "PTX/Tree/Instructions/Modifiers/RoundingModifier.h"
#include "PTX/Tree/Instructions/Modifiers/SaturateModifier.h"
#include "PTX/Tree/Instructions/Modifiers/ScopeModifier.h"
#include "PTX/Tree/Instructions/Modifiers/UniformModifier.h"

#include "PTX/Tree/Instructions/Arithmetic/AbsoluteInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/AddInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/BitFieldExtractInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/BitFieldInsertInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/BitFindInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/BitReverseInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/CopySignInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/CosineInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/CountLeadingZerosInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/DivideInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/Exp2Instruction.h"
#include "PTX/Tree/Instructions/Arithmetic/FindNthBitInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/FMAInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/Log2Instruction.h"
#include "PTX/Tree/Instructions/Arithmetic/MADInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/MADWideInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/MaximumInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/MultiplyInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/MultiplyWideInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/MinimumInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/NegateInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/PopulationCountInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/ReciprocalInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/ReciprocalRootInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/RemainderInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/RootInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/SADInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/SineInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/SubtractInstruction.h"
#include "PTX/Tree/Instructions/Arithmetic/TestPropertyInstruction.h"

#include "PTX/Tree/Instructions/Comparison/SelectInstruction.h"
#include "PTX/Tree/Instructions/Comparison/SetInstruction.h"
#include "PTX/Tree/Instructions/Comparison/SetPredicateInstruction.h"
#include "PTX/Tree/Instructions/Comparison/SignSelectInstruction.h"

#include "PTX/Tree/Instructions/ControlFlow/BranchInstruction.h"
#include "PTX/Tree/Instructions/ControlFlow/BranchIndexInstruction.h"
#include "PTX/Tree/Instructions/ControlFlow/CallInstructionBase.h"
#include "PTX/Tree/Instructions/ControlFlow/CallInstruction.h"
#include "PTX/Tree/Instructions/ControlFlow/ExitInstruction.h"
#include "PTX/Tree/Instructions/ControlFlow/ReturnInstruction.h"

#include "PTX/Tree/Instructions/Logical/AndInstruction.h"
#include "PTX/Tree/Instructions/Logical/CNotInstruction.h"
#include "PTX/Tree/Instructions/Logical/Logical3OpInstruction.h"
#include "PTX/Tree/Instructions/Logical/NotInstruction.h"
#include "PTX/Tree/Instructions/Logical/OrInstruction.h"
#include "PTX/Tree/Instructions/Logical/XorInstruction.h"

#include "PTX/Tree/Instructions/Synchronization/FenceInstruction.h"
#include "PTX/Tree/Instructions/Synchronization/MemoryBarrierInstruction.h"
#include "PTX/Tree/Instructions/Synchronization/ReductionInstruction.h"
#include "PTX/Tree/Instructions/Synchronization/VoteInstruction.h"
#include "PTX/Tree/Instructions/Synchronization/ActiveMaskInstruction.h"
#include "PTX/Tree/Instructions/Synchronization/BarrierWarpInstruction.h"
#include "PTX/Tree/Instructions/Synchronization/BarrierInstruction.h"
#include "PTX/Tree/Instructions/Synchronization/BarrierReductionInstruction.h"
#include "PTX/Tree/Instructions/Synchronization/MatchInstruction.h"
#include "PTX/Tree/Instructions/Synchronization/AtomicInstruction.h"

#include "PTX/Tree/Instructions/Data/Modifiers/ConvertFlushSubnormalModifier.h"
#include "PTX/Tree/Instructions/Data/Modifiers/ConvertRoundingModifier.h"
#include "PTX/Tree/Instructions/Data/Modifiers/ConvertSaturateModifier.h"

#include "PTX/Tree/Instructions/Data/ConvertAddressInstruction.h"
#include "PTX/Tree/Instructions/Data/ConvertInstruction.h"
#include "PTX/Tree/Instructions/Data/IsSpaceInstruction.h"
#include "PTX/Tree/Instructions/Data/LoadInstruction.h"
#include "PTX/Tree/Instructions/Data/LoadNCInstruction.h"
#include "PTX/Tree/Instructions/Data/LoadUniformInstruction.h"
#include "PTX/Tree/Instructions/Data/MoveAddressInstruction.h"
#include "PTX/Tree/Instructions/Data/MoveInstruction.h"
#include "PTX/Tree/Instructions/Data/PackInstruction.h"
#include "PTX/Tree/Instructions/Data/PermuteInstruction.h"
#include "PTX/Tree/Instructions/Data/PrefetchInstruction.h"
#include "PTX/Tree/Instructions/Data/PrefetchUniformInstruction.h"
#include "PTX/Tree/Instructions/Data/ShuffleInstruction.h"
#include "PTX/Tree/Instructions/Data/StoreInstruction.h"
#include "PTX/Tree/Instructions/Data/UnpackInstruction.h"

#include "PTX/Tree/Instructions/Shift/ShiftLeftInstruction.h"
#include "PTX/Tree/Instructions/Shift/ShiftRightInstruction.h"
#include "PTX/Tree/Instructions/Shift/FunnelShiftInstruction.h"
