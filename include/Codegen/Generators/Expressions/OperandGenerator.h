#pragma once

#include "HorseIR/Traversal/ForwardTraversal.h"

#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/Literal.h"
#include "HorseIR/Tree/Types/Type.h"
#include "HorseIR/Tree/Types/PrimitiveType.h"

#include "PTX/Instructions/Data/ConvertInstruction.h"
#include "PTX/Operands/Value.h"

#include "Codegen/Builder.h"

template<PTX::Bits B, class T>
class OperandGenerator : public HorseIR::ForwardTraversal
{
public:
	OperandGenerator(Builder *builder) : m_builder(builder) {}

	const PTX::Operand<T> *GenerateOperand(HorseIR::Expression *expression)
	{
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
		//TODO: Handle list and primitive types
		HorseIR::PrimitiveType *type = static_cast<HorseIR::PrimitiveType *>(m_builder->GetCurrentSymbolTable()->GetType(identifier->GetName()));
		switch (type->GetKind())
		{
			case HorseIR::PrimitiveType::Kind::Int8:
				GenerateOperand<PTX::Int8Type>(identifier);
				break;
			case HorseIR::PrimitiveType::Kind::Int16:
				GenerateOperand<PTX::Int16Type>(identifier);
				break;
			case HorseIR::PrimitiveType::Kind::Int32:
				GenerateOperand<PTX::Int32Type>(identifier);
				break;
			case HorseIR::PrimitiveType::Kind::Int64:
				GenerateOperand<PTX::Int64Type>(identifier);
				break;
			case HorseIR::PrimitiveType::Kind::Float32:
				GenerateOperand<PTX::Float32Type>(identifier);
				break;
			case HorseIR::PrimitiveType::Kind::Float64:
				GenerateOperand<PTX::Float64Type>(identifier);
				break;
			default:
				std::cerr << "[ERROR] Unsupported type " << type->ToString() << " in function " << m_builder->GetCurrentFunction()->GetName() << std::endl;
				std::exit(EXIT_FAILURE);
		}
	}

	template<class S>
	void GenerateOperand(HorseIR::Identifier *identifier)
	{
		if constexpr(std::is_same<T, S>::value)
		{
			m_operand = m_builder->GetCurrentResources()->template GetRegister<T>(identifier->GetName());
		}
		else
		{
			auto source = m_builder->GetCurrentResources()->template GetRegister<S>(identifier->GetName());
			auto converted = m_builder->GetCurrentResources()->template AllocateRegister<T, ResourceType::Temporary>(identifier->GetName());
			m_builder->AddStatement(new PTX::ConvertInstruction<T, S>(converted, source));
			m_operand = converted;
		}
	}

	void Visit(HorseIR::Literal<int64_t> *literal) override
	{
		if (literal->GetCount() == 1)
		{
			if constexpr(std::is_same<T, PTX::Int64Type>::value)
			{
				m_operand = new PTX::Value<T>(literal->GetValue(0));
			}
			else
			{
				//TODO: Convert type for other types
				m_operand = new PTX::Value<T>((typename T::SystemType)literal->GetValue(0));
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

	const PTX::Operand<T> *m_operand = nullptr;
};
