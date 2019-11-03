#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"

#include "HorseIR/Tree/Tree.h"

#include "Utils/Logger.h"

namespace Codegen {

template<class T>
class LiteralGenerator : public HorseIR::ConstVisitor
{
public:
	static const HorseIR::TypedVectorLiteral<T> *GetLiteral(const HorseIR::Operand *operand)
	{
		LiteralGenerator generator;
		operand->Accept(generator);
		return generator.m_literal;
	}

	void Visit(const HorseIR::BooleanLiteral *literal) override
	{
		VisitLiteral<char>(literal);
	}

	void Visit(const HorseIR::CharLiteral *literal) override
	{
		VisitLiteral<char>(literal);
	}

	void Visit(const HorseIR::Int8Literal *literal) override
	{
		VisitLiteral<std::int8_t>(literal);
	}

	void Visit(const HorseIR::Int16Literal *literal) override
	{
		VisitLiteral<std::int16_t>(literal);
	}

	void Visit(const HorseIR::Int32Literal *literal) override
	{
		VisitLiteral<std::int32_t>(literal);
	}

	void Visit(const HorseIR::Int64Literal *literal) override
	{
		VisitLiteral<std::int64_t>(literal);
	}

	void Visit(const HorseIR::Float32Literal *literal) override
	{
		VisitLiteral<float>(literal);
	}

	void Visit(const HorseIR::Float64Literal *literal) override
	{
		VisitLiteral<double>(literal);
	}

	template<class L>
	void VisitLiteral(const HorseIR::TypedVectorLiteral<L> *literal)
	{
		if constexpr(std::is_same<T, L>::value)
		{
			m_literal = literal;
		}
		else
		{
			Utils::Logger::LogError("Requested literal type does not match actual type");
		}
	}

private:
	const HorseIR::TypedVectorLiteral<T> *m_literal = nullptr;
};

}
