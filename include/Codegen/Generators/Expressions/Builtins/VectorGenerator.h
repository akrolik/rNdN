#pragma once

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B, class T, typename Enabled = void>
class VectorGenerator : public BuiltinGenerator<B, T>
{
public:
	using BuiltinGenerator<B, T>::BuiltinGenerator;
};

template<PTX::Bits B, class T>
class VectorGenerator<B, T, std::enable_if_t<PTX::MoveInstruction<T, false>::TypeSupported>> : public BuiltinGenerator<B, T>
{
public:
	using BuiltinGenerator<B, T>::BuiltinGenerator;

	const PTX::Register<T> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		OperandGenerator<B, T> opGen(this->m_builder);
		auto src = opGen.GenerateOperand(arguments.at(1), OperandGenerator<B, T>::LoadKind::Vector);

		auto targetRegister = this->GenerateTargetRegister(target, arguments);
		this->m_builder.AddStatement(new PTX::MoveInstruction<T>(targetRegister, src));

		return targetRegister;
	}
};

template<PTX::Bits B>
class VectorGenerator<B, PTX::Int8Type> : public BuiltinGenerator<B, PTX::Int8Type>
{
public:
	using BuiltinGenerator<B, PTX::Int8Type>::BuiltinGenerator;

	const PTX::Register<PTX::Int8Type> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		// With 8-bit data, we must use a packed move

		auto resources = this->m_builder.GetLocalResources();

		OperandGenerator<B, PTX::Int8Type> opGen(this->m_builder);
		auto src = opGen.GenerateOperand(arguments.at(1), OperandGenerator<B, PTX::Int8Type>::LoadKind::Vector);

		auto targetRegister = this->GenerateTargetRegister(target, arguments);

		auto temp = resources->template AllocateTemporary<PTX::Bit16Type>();
		auto bracedSource = new PTX::Braced2Operand<PTX::Bit8Type>({
			new PTX::Bit8Adapter<PTX::IntType>(src),
			new PTX::Value<PTX::Bit8Type>(0)
		});
		auto bracedTarget = new PTX::Braced2Register<PTX::Bit8Type>({
			new PTX::Bit8RegisterAdapter<PTX::IntType>(targetRegister),
			new PTX::SinkRegister<PTX::Bit8Type>
		});

		this->m_builder.AddStatement(new PTX::Pack2Instruction<PTX::Bit16Type>(temp, bracedSource));
		this->m_builder.AddStatement(new PTX::Unpack2Instruction<PTX::Bit16Type>(bracedTarget, temp));

		return targetRegister;
	}
};

}
