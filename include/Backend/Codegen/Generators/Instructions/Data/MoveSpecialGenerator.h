#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"
#include "PTX/Traversal/ConstOperandVisitor.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class MoveSpecialGenerator : public PredicatedInstructionGenerator, public PTX::ConstOperandVisitor
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "MoveSpecialGenerator"; }

	// Instruction

	void Generate(const PTX::_MoveSpecialInstruction *instruction);

	template<class T>
	void Visit(const PTX::MoveSpecialInstruction<T> *instruction);

	// Special Register

	bool Visit(const PTX::_SpecialRegister *reg) override;
	bool Visit(const PTX::_IndexedRegister *reg) override;

	template<class T>
	void Visit(const PTX::SpecialRegister<T> *reg);

	template<class T, class S, PTX::VectorSize V>
	void Visit(const PTX::IndexedRegister<T, S, V> *reg);

private:
	void GenerateS2R(SASS::SpecialRegister::Kind special);
	void GenerateMOV(std::uint32_t maxwellConstant, std::uint32_t voltaConstant);
	void GeneratePM64(SASS::SpecialRegister::Kind specialLo, SASS::SpecialRegister::Kind specialHi);

	SASS::Register *m_destination = nullptr;
	SASS::Register *m_destinationHi = nullptr;

	const PTX::InstructionStatement *m_instruction = nullptr;
};

}
}
