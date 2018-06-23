#pragma once

#include "HorseIR/Traversal/ForwardTraversal.h"

#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/Literal.h"
#include "HorseIR/Tree/Types/Type.h"
#include "HorseIR/Tree/Types/ListType.h"
#include "HorseIR/Tree/Types/PrimitiveType.h"

#include "PTX/Instructions/Data/ConvertInstruction.h"
#include "PTX/Operands/Value.h"

#include "Codegen/Builder.h"

template<PTX::Bits B, class T>
class OperandGenerator : public HorseIR::ForwardTraversal
{
public:
	OperandGenerator(Builder *builder) : m_builder(builder) {}

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

	void Visit(HorseIR::Identifier *identifier) override
	{
		HorseIR::Type *type = m_builder->GetCurrentSymbolTable()->GetType(identifier->GetString());
		switch (type->GetKind())
		{
			case HorseIR::Type::Kind::Primitive:
				Dispatch(static_cast<HorseIR::PrimitiveType *>(type), identifier);
				break;
			case HorseIR::Type::Kind::List:
				Dispatch(static_cast<HorseIR::ListType *>(type), identifier);
				break;
			default:
				std::cerr << "[ERROR] Unsupported type " << type->ToString() << " in function " << m_builder->GetCurrentFunction()->GetName() << std::endl;
				std::exit(EXIT_FAILURE);
		}
	}

	void Dispatch(HorseIR::PrimitiveType *type, HorseIR::Identifier *identifier)
	{
		switch (type->GetKind())
		{
			case HorseIR::PrimitiveType::Kind::Int8:
				GenerateIdentifier<PTX::Int8Type>(identifier);
				break;
			case HorseIR::PrimitiveType::Kind::Int16:
				GenerateIdentifier<PTX::Int16Type>(identifier);
				break;
			case HorseIR::PrimitiveType::Kind::Int32:
				GenerateIdentifier<PTX::Int32Type>(identifier);
				break;
			case HorseIR::PrimitiveType::Kind::Int64:
				GenerateIdentifier<PTX::Int64Type>(identifier);
				break;
			case HorseIR::PrimitiveType::Kind::Float32:
				GenerateIdentifier<PTX::Float32Type>(identifier);
				break;
			case HorseIR::PrimitiveType::Kind::Float64:
				GenerateIdentifier<PTX::Float64Type>(identifier);
				break;
			default:
				std::cerr << "[ERROR] Unsupported type " << type->ToString() << " in function " << m_builder->GetCurrentFunction()->GetName() << std::endl;
				std::exit(EXIT_FAILURE);
		}
	}

	void Dispatch(HorseIR::ListType *type, HorseIR::Identifier *identifier)
	{
		std::cerr << "[ERROR] Unsupported type " << type->ToString() << " in function " << m_builder->GetCurrentFunction()->GetName() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	template<class S>
	void GenerateIdentifier(HorseIR::Identifier *identifier)
	{
		if constexpr(std::is_same<T, S>::value)
		{
			m_operand = m_builder->GetCurrentResources()->template GetRegister<T>(identifier->GetString());
		}
		else
		{
			auto source = m_builder->GetCurrentResources()->template GetRegister<S>(identifier->GetString());
			auto converted = m_builder->GetCurrentResources()->template AllocateRegister<T, ResourceType::Temporary>(identifier->GetString());
			m_builder->AddStatement(new PTX::ConvertInstruction<T, S>(converted, source));
			m_operand = converted;
		}
	}

	void Visit(HorseIR::Literal<int64_t> *literal) override
	{
		GenerateLiteral<int64_t>(literal);
	}

	void Visit(HorseIR::Literal<double> *literal) override
	{
		GenerateLiteral<double>(literal);
	}

	template<class L>
	void GenerateLiteral(HorseIR::Literal<L> *literal)
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
	Builder *m_builder = nullptr;

	const PTX::TypedOperand<T> *m_operand = nullptr;
};
