#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"

#include "HorseIR/Tree/Tree.h"

#include "Utils/Logger.h"

namespace HorseIR {

template<class T>
class LiteralUtils : public ConstVisitor
{
public:
	static const TypedVectorLiteral<T> *GetLiteral(const Operand *operand)
	{
		LiteralUtils utils;
		operand->Accept(utils);
		return utils.m_literal;
	}

	void Visit(const BooleanLiteral *literal) override
	{
		VisitLiteral<std::int8_t>(literal);
	}

	void Visit(const CharLiteral *literal) override
	{
		VisitLiteral<std::int8_t>(literal);
	}

	void Visit(const Int8Literal *literal) override
	{
		VisitLiteral<std::int8_t>(literal);
	}

	void Visit(const Int16Literal *literal) override
	{
		VisitLiteral<std::int16_t>(literal);
	}

	void Visit(const Int32Literal *literal) override
	{
		VisitLiteral<std::int32_t>(literal);
	}

	void Visit(const Int64Literal *literal) override
	{
		VisitLiteral<std::int64_t>(literal);
	}

	void Visit(const Float32Literal *literal) override
	{
		VisitLiteral<float>(literal);
	}

	void Visit(const Float64Literal *literal) override
	{
		VisitLiteral<double>(literal);
	}

	void Visit(const StringLiteral *literal) override
	{
		VisitLiteral<std::string>(literal);
	}

	void Visit(const SymbolLiteral *literal) override
	{
		VisitLiteral<SymbolValue *>(literal);
	}

	void Visit(const DatetimeLiteral *literal) override
	{
		VisitLiteral<DatetimeValue *>(literal);
	}

	void Visit(const MonthLiteral *literal) override
	{
		VisitLiteral<MonthValue *>(literal);
	}

	void Visit(const DateLiteral *literal) override
	{
		VisitLiteral<DateValue *>(literal);
	}

	void Visit(const MinuteLiteral *literal) override
	{
		VisitLiteral<MinuteValue *>(literal);
	}

	void Visit(const SecondLiteral *literal) override
	{
		VisitLiteral<SecondValue *>(literal);
	}

	void Visit(const TimeLiteral *literal) override
	{
		VisitLiteral<TimeValue *>(literal);
	}

	template<class L>
	void VisitLiteral(const TypedVectorLiteral<L> *literal)
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
	const TypedVectorLiteral<T> *m_literal = nullptr;
};

}
