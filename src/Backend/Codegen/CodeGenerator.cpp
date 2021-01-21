#include "Backend/Codegen/CodeGenerator.h"

#include "Backend/Codegen/Generators/InstructionGenerator.h"

namespace Backend {
namespace Codegen {

// Public API

SASS::Function *CodeGenerator::Generate(const PTX::FunctionDefinition<PTX::VoidType> *function, const PTX::Analysis::RegisterAllocation *allocation)
{
	auto sassFunction = m_builder.CreateFunction(function->GetName(), allocation);
	function->Accept(*this);
	m_builder.CloseFunction();
	return sassFunction;
}

// Functions

void CodeGenerator::VisitOut(const PTX::FunctionDefinition<PTX::VoidType> *function)
{
	return;

	//TODO: Generate SASS program

	auto sassBlock0 = new SASS::BasicBlock("BB0");
	// m_function->AddBasicBlock(sassBlock0);

	// /*0008*/                   MOV R1, c[0x0][0x20] ;
	// /*0010*/                   S2R R4, SR_CTAID.X ;
	// /*0018*/                   S2R R2, SR_TID.X ;
	//
	// /*0028*/                   XMAD.MRG R3, R4.reuse, c[0x0] [0x8].H1, RZ ;
	// /*0030*/                   XMAD R2, R4.reuse, c[0x0] [0x8], R2 ;
	// /*0038*/                   XMAD.PSL.CBCC R4, R4.H1, R3.H1, R2 ;
	//
	// /*0048*/                   IADD R2.CC, R4, c[0x0][0x140] ; // t1
	// /*0050*/                   IADD.X R3, RZ, c[0x0][0x144] ;
	// /*0058*/                   LDG.E.U8 R2, [R2] ;
	// 
	// /*0068*/                   IADD R4.CC, R4, c[0x0][0x150] ; // $r0
	// /*0070*/                   IADD.X R5, RZ, c[0x0][0x154] ;
	// /*0078*/                   IADD32I R6, R2, 0x1 ;
	//
	// /*0088*/                   STG.E.U8 [R4], R6 ;
	// /*0090*/                   EXIT ;

	auto R1 = new SASS::Register(1);
	auto R2 = new SASS::Register(2);
	auto R3 = new SASS::Register(3);
	auto R4 = new SASS::Register(4);
	auto R5 = new SASS::Register(5);
	auto R6 = new SASS::Register(6);

	// 0x001cfc00e22007f6:
	//
	//   0000 000000 111 111 1 0110
	//   0000 000000 111 000 1 0001
	//   0000 000000 111 001 1 1111

	auto inst1 = new SASS::MOVInstruction(R1, new SASS::Constant(0x0, 0x20));
	auto inst2 = new SASS::S2RInstruction(R4, new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_CTAID_X));
	auto inst3 = new SASS::S2RInstruction(R2, new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_TID_X));

	inst1->SetScheduling(6, true, 7, 7, 0, 0);
	inst2->SetScheduling(1, true, 0, 7, 0, 0);
	inst3->SetScheduling(15, true, 1, 7, 0, 0);

	sassBlock0->AddInstruction(inst1);
	sassBlock0->AddInstruction(inst2);
	sassBlock0->AddInstruction(inst3);

	// 0x001fd842fec20ff1:
	//
	//   0001 000001 111 111 1 0001
	//   0001 000010 111 111 1 0110
	//   0000 000000 111 111 1 0110

	auto inst4 = new SASS::XMADInstruction(R3, R4, new SASS::Constant(0x0, 0x8), SASS::RZ,
			SASS::XMADInstruction::Mode::MRG, SASS::XMADInstruction::Flags::H1_B
	);
	auto inst5 = new SASS::XMADInstruction(R2, R4, new SASS::Constant(0x0, 0x8), R2);
	auto inst6 = new SASS::XMADInstruction(R4, R4, R3, R2,
		SASS::XMADInstruction::Mode::PSL,
		SASS::XMADInstruction::Flags::CBCC | SASS::XMADInstruction::Flags::H1_A | SASS::XMADInstruction::Flags::H1_B
	);

	inst4->SetScheduling(1, true, 7, 7, 1, 1);
	inst5->SetScheduling(6, true, 7, 7, 2, 1);
	inst6->SetScheduling(6, true, 7, 7, 0, 0);

	sassBlock0->AddInstruction(inst4);
	sassBlock0->AddInstruction(inst5);
	sassBlock0->AddInstruction(inst6);

	// 0x001ec400fc4007f6:
	//
	//   0000 000000 111 111 1 0110
	//   0000 000000 111 111 0 0010
	//   0000 000000 111 101 1 0001

	auto inst7 = new SASS::IADDInstruction(R2, R4, new SASS::Constant(0x0, 0x140), SASS::IADDInstruction::Flags::CC);
	auto inst8 = new SASS::IADDInstruction(R3, SASS::RZ, new SASS::Constant(0x0, 0x144), SASS::IADDInstruction::Flags::X);
	auto inst9 = new SASS::LDGInstruction(R2, new SASS::Address(R2), SASS::LDGInstruction::Type::U8, SASS::LDGInstruction::Cache::None, SASS::LDGInstruction::Flags::E);

	inst7->SetScheduling(6, true, 7, 7, 0, 0);
	inst8->SetScheduling(2, false, 7, 7, 0, 0);
	inst9->SetScheduling(1, true, 5, 7, 0, 0);

	sassBlock0->AddInstruction(inst7);
	sassBlock0->AddInstruction(inst8);
	sassBlock0->AddInstruction(inst9);

	// 0x041fc800fee007f6:
	//
	//   0000 000000 111 111 1 0110
	//   0000 000000 111 111 1 0111
	//   0000 100000 111 111 1 0010

	auto inst10 = new SASS::IADDInstruction(R4, R4, new SASS::Constant(0x0, 0x150), SASS::IADDInstruction::Flags::CC);
	auto inst11 = new SASS::IADDInstruction(R5, SASS::RZ, new SASS::Constant(0x0, 0x154), SASS::IADDInstruction::Flags::X);
	auto inst12 = new SASS::IADD32IInstruction(R6, R2, new SASS::I32Immediate(0x1));

	inst10->SetScheduling(6, true, 7, 7, 0, 0);
	inst11->SetScheduling(7, true, 7, 7, 0, 0);
	inst12->SetScheduling(2, true, 7, 7, 32, 0);

	sassBlock0->AddInstruction(inst10);
	sassBlock0->AddInstruction(inst11);
	sassBlock0->AddInstruction(inst12);

	// 0x001ffc00ffe007f1:
	//
	//   0000 000000 111 111 1 0001
	//   0000 000000 111 111 1 1111

	auto inst13 = new SASS::STGInstruction(new SASS::Address(R4), R6, SASS::STGInstruction::Type::U8, SASS::STGInstruction::Cache::None, SASS::STGInstruction::Flags::E);
	auto inst14 = new SASS::EXITInstruction();

	inst13->SetScheduling(1, true, 7, 7, 0, 0);

	sassBlock0->AddInstruction(inst13);
	sassBlock0->AddInstruction(inst14);
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
	if constexpr(std::is_same<S, PTX::ParameterSpace>::value)
	{
		m_builder.AddParameter(PTX::BitSize<T::TypeBits>::NumBytes);
	}
}

// Basic Block

bool CodeGenerator::VisitIn(const PTX::BasicBlock *block)
{
	m_builder.CreateBasicBlock(block->GetLabel()->GetName());
	return true;
}

void CodeGenerator::VisitOut(const PTX::BasicBlock *block)
{
	m_builder.CloseBasicBlock();
}

// Statements

bool CodeGenerator::VisitIn(const PTX::InstructionStatement *statement)
{
	InstructionGenerator generator(m_builder);
	statement->Accept(generator);
	return false;
}

}
}
