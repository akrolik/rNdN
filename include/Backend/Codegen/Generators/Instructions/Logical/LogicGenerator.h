#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/PredicateGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class LogicGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

protected:
	template<template<class> class I, class T>
	void GenerateLogicMaxwell(const I<T> *instruction, SASS::Maxwell::PSETPInstruction::BooleanOperator1 predicateOperator, SASS::Maxwell::LOPInstruction::BooleanOperator integerOperator)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			// Generate operands

			PredicateGenerator predicateGenerator(this->m_builder);
			auto destination = predicateGenerator.Generate(instruction->GetDestination()).first;
			auto [sourceA, sourceA_Not] = predicateGenerator.Generate(instruction->GetSourceA());
			auto [sourceB, sourceB_Not] = predicateGenerator.Generate(instruction->GetSourceB());

			// Flags

			auto flags = SASS::Maxwell::PSETPInstruction::Flags::None;
			if (sourceA_Not)
			{
				flags |= SASS::Maxwell::PSETPInstruction::Flags::NOT_A;
			}
			if (sourceB_Not)
			{
				flags |= SASS::Maxwell::PSETPInstruction::Flags::NOT_B;
			}

			// Generate instruction

			this->AddInstruction(new SASS::Maxwell::PSETPInstruction(
				destination, SASS::PT, sourceA, sourceB, SASS::PT,
				predicateOperator, SASS::Maxwell::PSETPInstruction::BooleanOperator2::AND, flags
			));
		}
		else
		{
			RegisterGenerator registerGenerator(this->m_builder);
			auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());
			auto [sourceA_Lo, sourceA_Hi] = registerGenerator.GeneratePair(instruction->GetSourceA());

			CompositeGenerator compositeGenerator(this->m_builder);
			auto [sourceB_Lo, sourceB_Hi] = compositeGenerator.GeneratePair(instruction->GetSourceB());

			this->AddInstruction(new SASS::Maxwell::LOPInstruction(destination_Lo, sourceA_Lo, sourceB_Lo, integerOperator));

			if constexpr(T::TypeBits == PTX::Bits::Bits64)
			{
				this->AddInstruction(new SASS::Maxwell::LOPInstruction(destination_Hi, sourceA_Hi, sourceB_Hi, integerOperator));
			}
		}
	}

	template<template<class> class I, class T, typename FP, typename FI>
	void GenerateLogicVolta(const I<T> *instruction, FP functionPredicate, FI functionInteger)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			// Generate operands

			PredicateGenerator predicateGenerator(this->m_builder);
			auto destination = predicateGenerator.Generate(instruction->GetDestination()).first;
			auto [sourceA, sourceA_Not] = predicateGenerator.Generate(instruction->GetSourceA());
			auto [sourceB, sourceB_Not] = predicateGenerator.Generate(instruction->GetSourceB());

			// Flags

			auto flags = SASS::Volta::PLOP3Instruction::Flags::None;
			if (sourceA_Not)
			{
				flags |= SASS::Volta::PLOP3Instruction::Flags::NOT_A;
			}
			if (sourceB_Not)
			{
				flags |= SASS::Volta::PLOP3Instruction::Flags::NOT_B;
			}

			// Generate instruction

			auto logicOperation = SASS::Volta::BinaryUtils::LogicOperation(functionPredicate);

			this->AddInstruction(new SASS::Volta::PLOP3Instruction(
				destination, SASS::PT, sourceA, sourceB, SASS::PT, 
				new SASS::I8Immediate(logicOperation), new SASS::I8Immediate(0x0), flags
			));
		}
		else
		{
			// Generate operands

			RegisterGenerator registerGenerator(this->m_builder);
			auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());
			auto [sourceA_Lo, sourceA_Hi] = registerGenerator.GeneratePair(instruction->GetSourceA());

			CompositeGenerator compositeGenerator(this->m_builder);
			compositeGenerator.SetImmediateSize(32);
			auto [sourceB_Lo, sourceB_Hi] = compositeGenerator.GeneratePair(instruction->GetSourceB());

			// Generate instruction

			auto logicOperation = SASS::Volta::BinaryUtils::LogicOperation(functionInteger);

			this->AddInstruction(new SASS::Volta::LOP3Instruction(
				destination_Lo, sourceA_Lo, sourceB_Lo, SASS::RZ, new SASS::I8Immediate(logicOperation), SASS::PT
			));

			if constexpr(T::TypeBits == PTX::Bits::Bits64)
			{
				this->AddInstruction(new SASS::Volta::LOP3Instruction(
					destination_Hi, sourceA_Hi, sourceB_Hi, SASS::RZ, new SASS::I8Immediate(logicOperation), SASS::PT
				));
			}
		}
	}
};

}
}
