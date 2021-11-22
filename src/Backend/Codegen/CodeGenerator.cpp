#include "Backend/Codegen/CodeGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"
#include "Backend/Codegen/Generators/InstructionGenerator.h"
#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"

#include "Utils/Logger.h"

namespace Backend {
namespace Codegen {

// Public API

SASS::Function *CodeGenerator::Generate(const PTX::FunctionDefinition<PTX::VoidType> *function, const PTX::Analysis::RegisterAllocation *registerAllocation, const PTX::Analysis::ParameterSpaceAllocation *parameterAllocation)
{
	// Setup codegen builder

	auto sassFunction = m_builder.CreateFunction(function->GetName());
	m_builder.SetRegisterAllocation(registerAllocation);
	m_builder.SetParameterSpaceAllocation(parameterAllocation);

	// Properties

	m_builder.SetMaxThreads(function->GetMaxThreads());
	m_builder.SetRequiredThreads(function->GetRequiredThreads());

	// Traverse function

	for (const auto& parameter : function->GetParameters())
	{
		parameter->Accept(static_cast<ConstHierarchicalVisitor&>(*this));
	}

	// Construct basic blocks

	function->GetStructuredGraph()->Accept(*this);

	// Close function and return

	m_builder.SetCRSStackSize(m_maxStack);
	m_builder.CloseFunction();
	return sassFunction;
}

// Declarations

bool CodeGenerator::VisitIn(const PTX::VariableDeclaration *declaration)
{
	declaration->Accept(static_cast<ConstDeclarationVisitor&>(*this));
	return false;
}

void CodeGenerator::Visit(const PTX::_TypedVariableDeclaration *declaration)
{
	declaration->Dispatch(*this);
}

template<class T, class S>
void CodeGenerator::Visit(const PTX::TypedVariableDeclaration<T, S> *declaration)
{
	for (const auto& name : declaration->GetNames())
	{
		for (auto i = 0u; i < name->GetCount(); ++i)
		{
			const auto string = name->GetName(i);
			const auto dataSize = PTX::BitSize<T::TypeBits>::NumBytes;

			if constexpr(std::is_same<S, PTX::ParameterSpace>::value)
			{
				// Add each parameter declaration to the parameter constant space

				m_builder.AddParameter(dataSize);
			}
			else if constexpr(std::is_same<S, PTX::SharedSpace>::value)
			{
				// Add each shared declaration to the function

				if constexpr(PTX::is_array_type<T>::value)
				{
					// Array sizes, only possible for shared spaces (not parameters)

					m_builder.AddSharedVariable(string, T::ElementCount * dataSize, dataSize);
				}
				else
				{
					m_builder.AddSharedVariable(string, dataSize, dataSize);
				}
			}
		}
	}
}

// Structure

void CodeGenerator::Visit(const PTX::Analysis::StructureNode *structure)
{
	// Generate next structure

	if (auto next = structure->GetNext())
	{
		next->Accept(*this);
	}
}

void CodeGenerator::Visit(const PTX::Analysis::BranchStructure *structure)
{
	ArchitectureDispatch::DispatchInline(m_builder, [&]()
	{
		GenerateBranchStructure<
			SASS::Maxwell::SSYInstruction,
			SASS::Maxwell::SYNCInstruction,
			SASS::Maxwell::BRAInstruction
		>(structure);
	},
	[&]()
	{
		GenerateBranchStructure<
			SASS::Volta::BSSYInstruction,
			SASS::Volta::BSYNCInstruction,
			SASS::Volta::BRAInstruction,
			true
		>(structure);
	});
}

void CodeGenerator::Visit(const PTX::Analysis::ExitStructure *structure)
{
	ArchitectureDispatch::DispatchInline(m_builder, [&]()
	{
		GenerateExitStructure<SASS::Maxwell::SYNCInstruction>(structure);
	},
	[&]()
	{
		GenerateExitStructure<SASS::Volta::BSYNCInstruction, true>(structure);
	});
}

void CodeGenerator::Visit(const PTX::Analysis::LoopStructure *structure)
{
	ArchitectureDispatch::DispatchInline(m_builder, [&]()
	{
		GenerateLoopStructure<
			SASS::Maxwell::SSYInstruction,
			SASS::Maxwell::BRAInstruction
		>(structure);
	},
	[&]()
	{
		GenerateLoopStructure<
			SASS::Volta::BSSYInstruction,
			SASS::Volta::BRAInstruction,
			true
		>(structure);
	});
}

void CodeGenerator::Visit(const PTX::Analysis::SequenceStructure *structure)
{
	// Generate current block

	structure->GetBlock()->Accept(*this);
	auto sequenceBlock = m_endBlock;

	// Process next structure

	PTX::Analysis::ConstStructuredGraphVisitor::Visit(structure);
	m_beginBlock = sequenceBlock;
}

template<class SSY, class SYNC, class BRA, bool BARRIER>
void CodeGenerator::GenerateBranchStructure(const PTX::Analysis::BranchStructure *structure)
{
	// Generate current block, including condition generation

	structure->GetBlock()->Accept(*this);
	auto branchBlock = m_endBlock;

	SSY *ssyInstruction = nullptr;
	SYNC *syncInstruction1 = nullptr;
	SYNC *syncInstruction2 = nullptr;

	if constexpr(BARRIER)
	{
		if (m_stackDepth > SASS::Volta::MAX_SSY_STACK_DEPTH)
		{
			Utils::Logger::LogError("Stack depth " + std::to_string(m_stackDepth) + " exceeded maximum (" + std::to_string(SASS::Volta::MAX_SSY_STACK_DEPTH) + ")");
		}

		ssyInstruction = new SSY("", m_stackDepth);
		syncInstruction1 = new SYNC(m_stackDepth);
		syncInstruction2 = new SYNC(m_stackDepth);
	}
	else
	{
		ssyInstruction = new SSY("");
		syncInstruction1 = new SYNC();
		syncInstruction2 = new SYNC();
	}
	branchBlock->AddInstruction(ssyInstruction);

	m_stackSize += SASS::SSY_STACK_SIZE;
	m_stackDepth++;
	if (m_stackSize > m_maxStack)
	{
		m_maxStack = m_stackSize;
	}

	// Generate the condition to predicate threads

	PredicateGenerator predicateGenerator(m_builder);
	auto [predicate, negate] = predicateGenerator.Generate(structure->GetPredicate()); 

	if (auto trueStructure = structure->GetTrueBranch())
	{
		BRA *branchInstruction = nullptr;

		if (auto falseStructure = structure->GetFalseBranch())
		{
			// If-Else pattern
			// 
			//     SSY L2
			//  @P BRA L1
			//     <False>
			//     SYNC
			// L1:
			//     <True>
			//     SYNC
			// L2:

			// Branch true threads

			branchInstruction = new BRA("");
			branchInstruction->SetPredicate(predicate, negate);
			branchBlock->AddInstruction(branchInstruction);

			// Generate false structure

			falseStructure->Accept(*this);

			// Sync false threads

			m_endBlock->AddInstruction(syncInstruction1);
		}
		else
		{
			// Only true branch pattern
			// 
			//     SSY L1
			// @!P SYNC
			//     <True>
			//     SYNC
			// L1:

			// Sync false threads

			syncInstruction1->SetPredicate(predicate, !negate);
			branchBlock->AddInstruction(syncInstruction1);
		}

		// Generate true structure

		trueStructure->Accept(*this);
		auto trueBlock = m_beginBlock;

		// Sync true threads

		m_endBlock->AddInstruction(syncInstruction2);

		// Patch true branch

		if (branchInstruction != nullptr)
		{
			branchInstruction->SetTarget(trueBlock->GetName());
		}
	}
	else if (auto falseStructure = structure->GetFalseBranch())
	{
		// Only false branch pattern
		// 
		//     SSY L1
		//  @P SYNC
		//     <False>
		//     SYNC
		// L1:

		// Sync true threads

		syncInstruction1->SetPredicate(predicate, negate);
		branchBlock->AddInstruction(syncInstruction1);

		// Generate false structure

		falseStructure->Accept(*this);

		// Sync false threads

		m_endBlock->AddInstruction(syncInstruction2);
	}

	m_stackSize -= SASS::SSY_STACK_SIZE;
	m_stackDepth--;

	// Process next structure

	PTX::Analysis::ConstStructuredGraphVisitor::Visit(structure);

	// Patch SSY reconvergence point

	const auto& reconvergencePoint = m_beginBlock->GetName();

	ssyInstruction->SetTarget(reconvergencePoint);
	m_builder.AddIndirectBranch(syncInstruction1, reconvergencePoint);
	m_builder.AddIndirectBranch(syncInstruction2, reconvergencePoint);

	m_beginBlock = branchBlock;
}

template<class SYNC, bool BARRIER>
void CodeGenerator::GenerateExitStructure(const PTX::Analysis::ExitStructure *structure)
{
	// Exit pattern
	//
	//     <Block>
	//  @P SYNC
	//     <Next>

	structure->GetBlock()->Accept(*this);
	auto block = m_endBlock;

	// Generate the condition to predicate threads

	auto [structurePredicate, structureNegate] = structure->GetPredicate();

	PredicateGenerator predicateGenerator(m_builder);
	auto [predicate, negate] = predicateGenerator.Generate(structurePredicate); 

	// Predicate threads for exit

	SYNC *syncInstruction = nullptr;
	if constexpr(BARRIER)
	{
		// Reduce by 1 to match SSY instruction

		syncInstruction = new SYNC(m_stackDepth - 1);
	}
	else
	{
		syncInstruction = new SYNC();
	}
	syncInstruction->SetPredicate(predicate, negate ^ structureNegate);

	m_loopExits.push_back(syncInstruction);
	m_endBlock->AddInstruction(syncInstruction);

	// Process next structure

	PTX::Analysis::ConstStructuredGraphVisitor::Visit(structure);

	m_beginBlock = block;
}

template<class SSY, class BRA, bool BARRIER>
void CodeGenerator::GenerateLoopStructure(const PTX::Analysis::LoopStructure *structure)
{
	// Loop pattern
	//
	//     SSY L2
	// L1:
	//     <Body>
	//     BRA L1
	// L2: 
	//     <Next>

	SSY *ssyInstruction = nullptr;
	if constexpr(BARRIER)
	{
		if (m_stackDepth > SASS::Volta::MAX_SSY_STACK_DEPTH)
		{
			Utils::Logger::LogError("Stack depth " + std::to_string(m_stackDepth) + " exceeded maximum (" + std::to_string(SASS::Volta::MAX_SSY_STACK_DEPTH) + ")");
		}

		ssyInstruction = new SSY("", m_stackDepth);
	}
	else
	{
		ssyInstruction = new SSY("");
	}
	m_endBlock->AddInstruction(ssyInstruction);

	m_stackSize += SASS::SSY_STACK_SIZE;
	m_stackDepth++;
	if (m_stackSize > m_maxStack)
	{
		m_maxStack = m_stackSize;
	}

	// Process loop body

	auto oldExits = m_loopExits;
	m_loopExits.clear();

	structure->GetBody()->Accept(*this);
	m_endBlock->AddInstruction(new BRA(m_beginBlock->GetName()));

	auto headerBlock = m_beginBlock;
	auto loopExits = m_loopExits;
	m_loopExits = oldExits;

	m_stackSize -= SASS::SSY_STACK_SIZE;
	m_stackDepth--;

	// Process next structure

	PTX::Analysis::ConstStructuredGraphVisitor::Visit(structure);

	// Patch header instruction and loop exit branches

	auto reconvergencePoint = m_beginBlock->GetName();
	ssyInstruction->SetTarget(reconvergencePoint);

	for (auto loopExit : loopExits)
	{
		m_builder.AddIndirectBranch(loopExit, reconvergencePoint);
	}

	m_beginBlock = headerBlock;
}

// Basic Block

bool CodeGenerator::VisitIn(const PTX::BasicBlock *block)
{
	m_builder.CreateBasicBlock(block->GetLabel()->GetName());
	return true;
}

void CodeGenerator::VisitOut(const PTX::BasicBlock *block)
{
	m_endBlock = m_builder.GetCurrentBlock();
	m_builder.CloseBasicBlock();
}

// Statements

bool CodeGenerator::VisitIn(const PTX::InstructionStatement *statement)
{
	// Clear all allocated temporary registers

	m_builder.ClearTemporaryRegisters();

	// Generate instruction

	InstructionGenerator generator(m_builder);
	generator.Generate(statement);
	return false;
}

}
}
