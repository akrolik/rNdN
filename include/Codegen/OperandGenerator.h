#pragma once

#include "HorseIR/Traversal/ForwardTraversal.h"

#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Expressions/Literal.h"

#include "PTX/Operands/Value.h"

#include "Codegen/ResourceAllocator.h"

template<PTX::Bits B, class T>
class OperandGenerator : public HorseIR::ForwardTraversal
{
public:
	OperandGenerator(ResourceAllocator<B> *resources, PTX::Function *function) : m_resources(resources), m_currentFunction(function) {}

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
		m_operand = m_resources->template GetRegister<T>(identifier->GetName());
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
	ResourceAllocator<B> *m_resources = nullptr;
	PTX::Function *m_currentFunction = nullptr;

	const PTX::Operand<T> *m_operand = nullptr;
};
