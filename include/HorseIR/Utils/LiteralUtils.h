#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"

#include "HorseIR/Tree/Tree.h"

#include "Utils/Logger.h"

namespace HorseIR {

template<class T>
class LiteralUtils : public HorseIR::ConstVisitor
{
public:
	static const HorseIR::TypedVectorLiteral<T> *GetLiteral(const HorseIR::Operand *operand)
	{
		LiteralUtils utils;
		operand->Accept(utils);
		return utils.m_literal;
	}

	void Visit(const HorseIR::BooleanLiteral *literal) override
	{
		VisitLiteral<std::int8_t>(literal);
	}

	void Visit(const HorseIR::CharLiteral *literal) override
	{
		VisitLiteral<std::int8_t>(literal);
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

	void Visit(const HorseIR::StringLiteral *literal) override
	{
		VisitLiteral<std::string>(literal);
	}

	void Visit(const HorseIR::SymbolLiteral *literal) override
	{
		VisitLiteral<HorseIR::SymbolValue *>(literal);
	}

	void Visit(const HorseIR::DatetimeLiteral *literal) override
	{
		VisitLiteral<HorseIR::DatetimeValue *>(literal);
	}

	void Visit(const HorseIR::MonthLiteral *literal) override
	{
		VisitLiteral<HorseIR::MonthValue *>(literal);
	}

	void Visit(const HorseIR::DateLiteral *literal) override
	{
		VisitLiteral<HorseIR::DateValue *>(literal);
	}

	void Visit(const HorseIR::MinuteLiteral *literal) override
	{
		VisitLiteral<HorseIR::MinuteValue *>(literal);
	}

	void Visit(const HorseIR::SecondLiteral *literal) override
	{
		VisitLiteral<HorseIR::SecondValue *>(literal);
	}

	void Visit(const HorseIR::TimeLiteral *literal) override
	{
		VisitLiteral<HorseIR::TimeValue *>(literal);
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
