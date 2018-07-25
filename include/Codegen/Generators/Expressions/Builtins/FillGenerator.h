#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "PTX/Instructions/Data/MoveInstruction.h"
#include "PTX/Instructions/Data/PackInstruction.h"
#include "PTX/Instructions/Data/UnpackInstruction.h"
#include "PTX/Operands/Adapters/BitAdapter.h"
#include "PTX/Operands/Variables/SinkRegister.h"
#include "PTX/Statements/BlockStatement.h"

namespace Codegen {

template<PTX::Bits B, class T, typename Enabled = void>
class FillGenerator : public BuiltinGenerator<B, T>
{
public:
	using BuiltinGenerator<B, T>::BuiltinGenerator;
};

template<PTX::Bits B, class T>
class FillGenerator<B, T, std::enable_if_t<PTX::MoveInstruction<T, false>::TypeSupported>> : public BuiltinGenerator<B, T>
{
public:
	using BuiltinGenerator<B, T>::BuiltinGenerator;

	void Generate(const PTX::Register<T> *target, const HorseIR::CallExpression *call) override
	{
		OperandGenerator<B, T> opGen(this->m_builder);
		auto src = opGen.GenerateOperand(call->GetArgument(1));
		this->m_builder->AddStatement(new PTX::MoveInstruction<T>(target, src));
	}
};

template<PTX::Bits B>
class FillGenerator<B, PTX::Int8Type> : public BuiltinGenerator<B, PTX::Int8Type>
{
public:
	using BuiltinGenerator<B, PTX::Int8Type>::BuiltinGenerator;

	void Generate(const PTX::Register<PTX::Int8Type> *target, const HorseIR::CallExpression *call) override
	{
		auto block = new PTX::BlockStatement();
		this->m_builder->AddStatement(block);
		this->m_builder->OpenScope(block);

		OperandGenerator<B, PTX::Int8Type> opGen(this->m_builder);
		auto src = opGen.GenerateOperand(call->GetArgument(1));

		auto temp = this->m_builder->template AllocateTemporary<PTX::Bit16Type>();
		auto bracedSource = new PTX::Braced2Operand<PTX::Bit8Type>({
			new PTX::Bit8Adapter<PTX::IntType>(src),
			new PTX::Value<PTX::Bit8Type>(0)
		});
		auto bracedTarget = new PTX::Braced2Register<PTX::Bit8Type>({
			new PTX::Bit8RegisterAdapter<PTX::IntType>(target),
			new PTX::SinkRegister<PTX::Bit8Type>
		});

		this->m_builder->AddStatement(new PTX::Pack2Instruction<PTX::Bit16Type>(temp, bracedSource));
		this->m_builder->AddStatement(new PTX::Unpack2Instruction<PTX::Bit16Type>(bracedTarget, temp));

		this->m_builder->CloseScope();
	}
};

}
