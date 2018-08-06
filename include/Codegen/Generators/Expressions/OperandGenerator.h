#pragma once

#include "HorseIR/Traversal/ForwardTraversal.h"
#include "Codegen/Generators/Generator.h"

#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/Literal.h"
#include "HorseIR/Tree/Types/Type.h"

#include "PTX/Instructions/Data/MoveInstruction.h"
#include "PTX/Instructions/Data/PackInstruction.h"
#include "PTX/Instructions/Data/UnpackInstruction.h"
#include "PTX/Operands/BracedOperand.h"
#include "PTX/Operands/Value.h"
#include "PTX/Operands/Adapters/BitAdapter.h"
#include "PTX/Operands/Variables/BracedRegister.h"
#include "PTX/Operands/Variables/SinkRegister.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/TypeDispatch.h"
#include "Codegen/Generators/Expressions/ConversionGenerator.h"

#include "Utils/Logger.h"

namespace Codegen {

template<PTX::Bits B, class T>
class OperandGenerator : public HorseIR::ForwardTraversal, public Generator
{
public:
	using Generator::Generator;

	const PTX::TypedOperand<T> *GenerateOperand(HorseIR::Expression *expression)
	{
		m_operand = nullptr;
		expression->Accept(*this);
		if (m_operand != nullptr)
		{
			return m_operand;
		}
		Utils::Logger::LogError("Unable to generate operand " + expression->ToString());
	}

	const PTX::Register<T> *GenerateRegister(HorseIR::Expression *expression)
	{
		const PTX::TypedOperand<T> *operand = GenerateOperand(expression);
		if (m_register)
		{
			return static_cast<const PTX::Register<T> *>(operand);
		}

		auto resources = this->m_builder.GetLocalResources();

		auto reg = resources->template AllocateTemporary<T>();
		if constexpr(std::is_same<T, PTX::Int8Type>::value)
		{
			auto bracedSource = new PTX::Braced2Operand<PTX::Bit8Type>({
				new PTX::Bit8Adapter<PTX::IntType>(operand),
				new PTX::Value<PTX::Bit8Type>(0)
			});
			auto bracedTarget = new PTX::Braced2Register<PTX::Bit8Type>({
				new PTX::Bit8RegisterAdapter<PTX::IntType>(reg),
				new PTX::SinkRegister<PTX::Bit8Type>
			});
			auto temp = resources->template AllocateTemporary<PTX::Bit16Type>();

			this->m_builder.AddStatement(new PTX::Pack2Instruction<PTX::Bit16Type>(temp, bracedSource));
			this->m_builder.AddStatement(new PTX::Unpack2Instruction<PTX::Bit16Type>(bracedTarget, temp));
		}
		else
		{
			this->m_builder.AddStatement(new PTX::MoveInstruction<T>(reg, operand));
		}
		return reg;
	}

	void Visit(HorseIR::Identifier *identifier) override
	{
		Codegen::DispatchType(*this, identifier->GetType(), identifier);
	}

	template<class S>
	void Generate(const HorseIR::Identifier *identifier)
	{
		auto resources = this->m_builder.GetLocalResources();
		if constexpr(std::is_same<T, S>::value)
		{
			m_operand = resources->GetRegister<T>(identifier->GetString());
		}
		else
		{
			auto source = resources->GetRegister<S>(identifier->GetString());
			m_operand = ConversionGenerator::ConvertSource<T, S>(this->m_builder, source);
		}
		m_register = true;
	}

	void Visit(HorseIR::Literal<int64_t> *literal) override
	{
		Generate<int64_t>(literal);
	}

	void Visit(HorseIR::Literal<double> *literal) override
	{
		Generate<double>(literal);
	}

	template<class L>
	void Generate(const HorseIR::Literal<L> *literal)
	{
		if (literal->GetCount() == 1)
		{
			if constexpr(std::is_same<typename T::SystemType, L>::value)
			{
				m_operand = new PTX::Value<T>(literal->GetValue(0));
			}
			else
			{
				m_operand = new PTX::Value<T>(static_cast<typename T::SystemType>(literal->GetValue(0)));
			}
		}
		else
		{
			Utils::Logger::LogError("[ERROR] Unsupported literal count: " + std::to_string(literal->GetCount()));
		}
	}

private:
	const PTX::TypedOperand<T> *m_operand = nullptr;
	bool m_register = false;
};

}
