#pragma once

#include <tuple>

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/OperandCompressionGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

enum class DateOperation {
	// Date
	Year,
	Month,
	Day,

	// Time
	Hour,
	Minute,
	Second,
	Millisecond
};

static std::string DateOperationString(DateOperation dateOp)
{
	switch (dateOp)
	{
		case DateOperation::Year:
			return "date_year";
		case DateOperation::Month:
			return "date_month";
		case DateOperation::Day:
			return "date_day";
		case DateOperation::Hour:
			return "time_hour";
		case DateOperation::Minute:
			return "time_minute";
		case DateOperation::Second:
			return "time_second";
		case DateOperation::Millisecond:
			return "time_mill";
	}
	return "<unknown>";
}

template<PTX::Bits B, class T, typename Enable = void>
class DateGenerator : public BuiltinGenerator<B, T>
{
public:
	DateGenerator(Builder& builder, DateOperation dateOp) : BuiltinGenerator<B, T>(builder), m_dateOp(dateOp) {}

	std::string Name() const override { return "DateGenerator"; }

private:
	DateOperation m_dateOp;
};

template<PTX::Bits B>
class DateGenerator<B, PTX::Int16Type> : public BuiltinGenerator<B, PTX::Int16Type>
{
	constexpr static auto UNIX_YEAR_BASE = 1970;

	constexpr static auto MONTHS_YEAR = 12;

	constexpr static auto DAYS_YEAR = 365;
	constexpr static auto DAYS_LYEAR = 366;

	constexpr static auto SECONDS_PER_MINUTE = 60;
	constexpr static auto MINUTES_PER_HOUR = 60;
	constexpr static auto HOURS_PER_DAY = 24;
	constexpr static auto SECONDS_PER_DAY = SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY;

public:
	DateGenerator(Builder& builder, DateOperation dateOp) : BuiltinGenerator<B, PTX::Int16Type>(builder), m_dateOp(dateOp) {}

	std::string Name() const override { return "DateGenerator"; }

	const PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<HorseIR::Operand *>& arguments) override
	{
		return OperandCompressionGenerator::UnaryCompressionRegister(this->m_builder, arguments);
	}

	const PTX::Register<PTX::Int16Type> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		OperandGenerator<B, PTX::Int32Type> opGen(this->m_builder);
		auto src = opGen.GenerateOperand(arguments.at(0), OperandGenerator<B, PTX::Int32Type>::LoadKind::Vector);

		auto targetRegister = this->GenerateTargetRegister(target, arguments);
		Generate(targetRegister, src);

		return targetRegister;
	}

	void Generate(const PTX::Register<PTX::Int16Type> *target, const PTX::TypedOperand<PTX::Int32Type> *src)
	{
		switch (m_dateOp)
		{
			case DateOperation::Year:
			{
				GenerateYear(target, src);
				break;
			}
			case DateOperation::Month:
			{
				GenerateMonth(target, src);
				break;
			}
			case DateOperation::Day:
			{
				GenerateDay(target, src);
				break;
			}
			default:
			{
				BuiltinGenerator<B, PTX::Int16Type>::Unimplemented("date operation " + DateOperationString(m_dateOp));
			}
		}
	}

	std::tuple<const PTX::Register<PTX::Int32Type> *, const PTX::Register<PTX::PredicateType> *> GenerateYear(
		const PTX::Register<PTX::Int16Type> *year, const PTX::TypedOperand<PTX::Int32Type> *src
	) {
		// Extract year from unix time
		//
		// days = time / SECONDS_PER_DAY
		// year = UNIX_YEAR_BASE
		//
		// while (true) {
		//     year_days = leap(year) ? DAYS_LYEAR : DAYS_YEAR
		//     if (days < year_days) break;
		//
		//     year++
		//     days -= year_days
		// }
		//
		// Leap year occurs every year:
		//   (1) Divisible by 4, but not by 100
		//   (2) Divisible by 400

		auto resources = this->m_builder.GetLocalResources();
		auto days = resources->template AllocateTemporary<PTX::Int32Type>();

		this->m_builder.AddStatement(new PTX::DivideInstruction<PTX::Int32Type>(days, src, new PTX::Int32Value(SECONDS_PER_DAY)));
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int16Type>(year, new PTX::Int16Value(UNIX_YEAR_BASE)));

		auto startLabel = this->m_builder.CreateLabel("START");
		auto endLabel = this->m_builder.CreateLabel("END");
		this->m_builder.AddStatement(startLabel);

		auto leapPredicate_4 = resources->template AllocateTemporary<PTX::PredicateType>();
		auto leapPredicate_100 = resources->template AllocateTemporary<PTX::PredicateType>();
		auto leapPredicate_400 = resources->template AllocateTemporary<PTX::PredicateType>();

		auto rem_4 = resources->template AllocateTemporary<PTX::Int16Type>();
		auto rem_100 = resources->template AllocateTemporary<PTX::Int16Type>();
		auto rem_400 = resources->template AllocateTemporary<PTX::Int16Type>();

		this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::Int16Type>(rem_4, year, new PTX::Int16Value(4)));
		this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::Int16Type>(rem_100, year, new PTX::Int16Value(100)));
		this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::Int16Type>(rem_400, year, new PTX::Int16Value(400)));

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::Int16Type>(
			leapPredicate_4, rem_4, new PTX::Int16Value(0), PTX::Int16Type::ComparisonOperator::Equal
		));
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::Int16Type>(
			leapPredicate_100, rem_100, new PTX::Int16Value(0), PTX::Int16Type::ComparisonOperator::NotEqual
		));
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::Int16Type>(
			leapPredicate_400, rem_400, new PTX::Int16Value(0), PTX::Int16Type::ComparisonOperator::Equal
		));

		auto tempPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		auto leapPredicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(tempPredicate, leapPredicate_4, leapPredicate_100));
		this->m_builder.AddStatement(new PTX::OrInstruction<PTX::PredicateType>(leapPredicate, tempPredicate, leapPredicate_400));

		auto yearDays = resources->template AllocateTemporary<PTX::Int32Type>();
		this->m_builder.AddStatement(new PTX::SelectInstruction<PTX::Int32Type>(
			yearDays, new PTX::Int32Value(DAYS_LYEAR), new PTX::Int32Value(DAYS_YEAR), leapPredicate
		));

		auto endPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::Int32Type>(
			endPredicate, days, yearDays, PTX::Int32Type::ComparisonOperator::Less
		));
		this->m_builder.AddStatement(new PTX::BranchInstruction(endLabel, endPredicate));

		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::Int16Type>(year, year, new PTX::Int16Value(1)));
		this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::Int32Type>(days, days, yearDays));
		this->m_builder.AddStatement(new PTX::BranchInstruction(startLabel));
		this->m_builder.AddStatement(endLabel);

		return {days, leapPredicate};
	}

	const PTX::Register<PTX::Int32Type> *GenerateMonth(const PTX::Register<PTX::Int16Type> *month, const PTX::TypedOperand<PTX::Int32Type> *src)
	{
		// Extract month from unix time
		//
		// [days, leap] = Year(year, src)
		//
		// month = 0
		// while (month < 12) {
		//     if (month == 1 && leap) {
		//         mdays = 29
		//     } else {
		//         mdays = const_days[month]
		//     }
		//
		//     if (days < mdays) {
		//         break
		//     }
		//     days -= mdays;
		// }
		// month += 1

		auto resources = this->m_builder.GetLocalResources();
		auto year = resources->template AllocateTemporary<PTX::Int16Type>();

		auto [days, leapPredicate] = GenerateYear(year, src);

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int16Type>(month, new PTX::Int16Value(0)));

		auto startLabel = this->m_builder.CreateLabel("START");
		auto endLabel = this->m_builder.CreateLabel("END");
		this->m_builder.AddStatement(startLabel);

		auto febPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		auto febLeapPredicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::Int16Type>(
			febPredicate, month, new PTX::Int16Value(1), PTX::Int16Type::ComparisonOperator::Equal
		));
		this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(febLeapPredicate, febPredicate, leapPredicate));

		// Get month days from constant space

		std::vector<std::int32_t> mdays({31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31});;

		auto moduleResources = this->m_builder.GetGlobalResources();
		auto c_months = new PTX::ArrayVariableAdapter<PTX::Int32Type, MONTHS_YEAR, PTX::ConstSpace>(
			moduleResources->template AllocateConstVariable<PTX::ArrayType<PTX::Int32Type, MONTHS_YEAR>>("mdays", mdays)
		);

		auto index = ConversionGenerator::ConvertSource<PTX::UInt32Type>(this->m_builder, month);

		AddressGenerator<B, PTX::Int32Type> addressGenerator(this->m_builder);
		auto c_monthAddress = addressGenerator.GenerateAddress(c_months, index);

		auto monthDays = resources->template AllocateTemporary<PTX::Int32Type>();
		this->m_builder.AddStatement(new PTX::LoadInstruction<B, PTX::Int32Type, PTX::ConstSpace>(monthDays, c_monthAddress));

		auto febMove = new PTX::MoveInstruction<PTX::Int32Type>(monthDays, new PTX::Int32Value(29));
		febMove->SetPredicate(febLeapPredicate);
		this->m_builder.AddStatement(febMove);

		// Increment before, since the +1 value is used in the next iteration or on exit

		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::Int16Type>(month, month, new PTX::Int16Value(1)));

		// Check if contained within the current month

		auto endPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::Int32Type>(
			endPredicate, days, monthDays, PTX::Int32Type::ComparisonOperator::Less
		));
		this->m_builder.AddStatement(new PTX::BranchInstruction(endLabel, endPredicate));

		this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::Int32Type>(days, days, monthDays));
		this->m_builder.AddStatement(new PTX::BranchInstruction(startLabel));
		this->m_builder.AddStatement(endLabel);

		return days;
	}

	void GenerateDay(const PTX::Register<PTX::Int16Type> *day, const PTX::TypedOperand<PTX::Int32Type> *src)
	{
		// Extract day from unix time
		//
		// days = Month(month, src)
		// days += 1

		auto resources = this->m_builder.GetLocalResources();
		auto month = resources->template AllocateTemporary<PTX::Int16Type>();

		auto days = GenerateMonth(month, src);
		auto days16 = ConversionGenerator::ConvertSource<PTX::Int16Type>(this->m_builder, days);

		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::Int16Type>(day, days16, new PTX::Int16Value(1)));
	}

	DateOperation m_dateOp;
};

}
