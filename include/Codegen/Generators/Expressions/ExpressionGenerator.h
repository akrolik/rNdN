#pragma once

#include "Codegen/Generators/Expressions/ExpressionGeneratorBase.h"

#include "PTX/Statements/BlockStatement.h"

#include "PTX/Instructions/DevInstruction.h"
#include "PTX/Instructions/Arithmetic/AddInstruction.h"
#include "PTX/Instructions/Data/ConvertInstruction.h"
#include "PTX/Instructions/Data/MoveInstruction.h"
#include "PTX/Instructions/Data/PackInstruction.h"
#include "PTX/Instructions/Data/UnpackInstruction.h"

#include "PTX/Operands/BracedOperand.h"
#include "PTX/Operands/Adapters/BitAdapter.h"
#include "PTX/Operands/Variables/BracedRegister.h"
#include "PTX/Operands/Variables/SinkRegister.h"

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
	
	void GenerateAdd(const PTX::TypedOperand<PTX::Int8Type> *src1, const PTX::TypedOperand<PTX::Int8Type> *src2) override
	{
		auto block = new PTX::BlockStatement();
		auto resources = this->m_builder->OpenScope(block);

		auto temp0 = resources->template AllocateRegister<PTX::Int16Type, ResourceType::Temporary>("0");
		auto temp1 = resources->template AllocateRegister<PTX::Int16Type, ResourceType::Temporary>("1");
		auto temp2 = resources->template AllocateRegister<PTX::Int16Type, ResourceType::Temporary>("2");

		block->AddStatement(new PTX::ConvertInstruction<PTX::Int16Type, PTX::Int8Type>(temp0, src1));
		block->AddStatement(new PTX::ConvertInstruction<PTX::Int16Type, PTX::Int8Type>(temp1, src2));
		block->AddStatement(new PTX::AddInstruction<PTX::Int16Type>(temp2, temp0, temp1));
		block->AddStatement(new PTX::ConvertInstruction<PTX::Int8Type, PTX::Int16Type>(this->m_target, temp2));

                this->m_builder->CloseScope();
		this->m_builder->AddStatement(block);
	}

	void GenerateMove(const PTX::TypedOperand<PTX::Int8Type> *src) override
	{
		auto block = new PTX::BlockStatement();
		auto resources = this->m_builder->OpenScope(block);

		auto temp = resources->template AllocateRegister<PTX::Bit16Type, ResourceType::Temporary>("0");
		auto value = new PTX::Value<PTX::Bit8Type>(0);

		auto bracedSource = new PTX::Braced2Operand<PTX::Bit8Type>({new PTX::Bit8Adapter<PTX::IntType>(src), value});
		auto bracedTarget = new PTX::Braced2Register<PTX::Bit8Type>({new PTX::Bit8RegisterAdapter<PTX::IntType>(this->m_target), new PTX::SinkRegister<PTX::Bit8Type>});

		block->AddStatement(new PTX::Pack2Instruction<PTX::Bit16Type>(temp, bracedSource));
		block->AddStatement(new PTX::Unpack2Instruction<PTX::Bit16Type>(bracedTarget, temp));

		this->m_builder->CloseScope();
		this->m_builder->AddStatement(block);
	}
};
