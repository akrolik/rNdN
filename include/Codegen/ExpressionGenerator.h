#pragma once

#include "Codegen/ExpressionGeneratorBase.h"

#include "PTX/Instructions/DevInstruction.h"

template<PTX::Bits B, class T>
class ExpressionGenerator : public ExpressionGeneratorBase<B, T> {
public:
	using ExpressionGeneratorBase<B, T>::ExpressionGeneratorBase;
};

template<PTX::Bits B>
class ExpressionGenerator<B, PTX::Int8Type> : public ExpressionGeneratorBase<B, PTX::Int8Type>
{
public:
	using ExpressionGeneratorBase<B, PTX::Int8Type>::ExpressionGeneratorBase;
	
	void GenerateAdd(const PTX::Operand<PTX::Int8Type> *src1, const PTX::Operand<PTX::Int8Type> *src2) override
	{
		//TODO: Update to use actual instuctions, decide on temporaries allocation

		this->m_currentFunction->AddStatement(new PTX::DevInstruction(".reg .s16 %tmp_a1, %tmp_a2, %tmp_a3"));
		this->m_currentFunction->AddStatement(new PTX::DevInstruction("cvt.s16.s8 %tmp_a1, " + src1->ToString()));
		this->m_currentFunction->AddStatement(new PTX::DevInstruction("cvt.s16.s8 %tmp_a2, " + src2->ToString()));
		this->m_currentFunction->AddStatement(new PTX::DevInstruction("add.s16 %tmp_a3, %tmp_a1, %tmp_a2"));
		this->m_currentFunction->AddStatement(new PTX::DevInstruction("cvt.s8.s16 " + this->m_target->ToString() + ", %tmp_a3"));
	}

	void GenerateMove(const PTX::Operand<PTX::Int8Type> *src) override
	{
		//TODO: Update to use actual instuctions, decice on temporary allocation

		std::string temp = this->m_target->ToString() + "_tmp";

		this->m_currentFunction->AddStatement(new PTX::DevInstruction(".reg .b16 " + temp));
		this->m_currentFunction->AddStatement(new PTX::DevInstruction("mov.b16 " + temp + ", {" + src->ToString() + ", 0}"));
		this->m_currentFunction->AddStatement(new PTX::DevInstruction("mov.b16 {" + this->m_target->ToString() + ", _}, " + temp));
	}
};
