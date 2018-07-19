#pragma once

#include "HorseIR/Traversal/ForwardTraversal.h"
#include "Codegen/Generators/Generator.h"

#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/Literal.h"
#include "HorseIR/Tree/Types/Type.h"

#include "PTX/Instructions/Data/ConvertInstruction.h"
#include "PTX/Instructions/Data/MoveInstruction.h"
#include "PTX/Operands/Value.h"
#include "PTX/Operands/Adapters/BitAdapter.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/TypeDispatch.h"

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

		std::cerr << "[ERROR] Unable to generate operand " << expression->ToString() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	const PTX::Register<T> *GenerateRegister(HorseIR::Expression *expression)
	{
		const PTX::TypedOperand<T> *operand = GenerateOperand(expression);
		if (m_register)
		{
			return static_cast<const PTX::Register<T> *>(operand);
		}

		auto reg = this->m_builder->template AllocateTemporary<T>();
		this->m_builder->AddStatement(new PTX::MoveInstruction<T>(reg, operand));

		return reg;
	}

	void Visit(HorseIR::Identifier *identifier) override
	{
		Codegen::DispatchType(*this, identifier->GetType(), identifier);
	}

	template<class S>
	void Generate(const HorseIR::Identifier *identifier)
	{
		if constexpr(std::is_same<T, S>::value)
		{
			m_operand = this->m_builder->GetRegister<T>(identifier->GetString());
		}
		else
		{
			//TODO: Implement conversion matrix, think about having destination and source generators
			auto source = this->m_builder->GetRegister<S>(identifier->GetString());
			if constexpr(std::is_same<PTX::BitType<S::TypeBits>, T>::value)
			{
				if constexpr(PTX::is_type_specialization<S, PTX::FloatType>::value)
				{
					m_operand = new PTX::BitRegisterAdapter<PTX::FloatType, S::TypeBits>(source);
				}
				else if constexpr(PTX::is_type_specialization<S, PTX::IntType>::value)
				{
					m_operand = new PTX::BitRegisterAdapter<PTX::IntType, S::TypeBits>(source);
				}
				else if constexpr(PTX::is_type_specialization<S, PTX::UIntType>::value)
				{
					m_operand = new PTX::BitRegisterAdapter<PTX::UIntType, S::TypeBits>(source);
				}
			}
			else if constexpr(PTX::ConvertInstruction<T, S, false>::TypeSupported)
			{
				auto converted = this->m_builder->AllocateTemporary<T>();
				this->m_builder->AddStatement(new PTX::ConvertInstruction<T, S>(converted, source));
				m_operand = converted;
			}
		}
		if (m_operand == nullptr)
		{
			std::cerr << "[ERROR] Unable to convert type " + S::Name() + " to type " + T::Name() << std::endl;
			std::exit(EXIT_FAILURE);
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
			std::cerr << "[ERROR] Unsupported literal count (>1)" << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}

private:
	const PTX::TypedOperand<T> *m_operand = nullptr;
	bool m_register = false;
};

}
