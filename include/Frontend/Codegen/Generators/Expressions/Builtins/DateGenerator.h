#pragma once

#include <tuple>

#include "Frontend/Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Expressions/OperandCompressionGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/OperandGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

#include "Utils/Date.h"

namespace Frontend {
namespace Codegen {

enum class DateOperation {
	// Date
	Date,                // s64     -> s32
	DateYear,            // s64,s32 -> s16
	DateMonth,           // s64,s32 -> s16
	DateDay,             // s64,s32 -> s16

	// Time
	Time,                // s64     -> s64
	TimeHour,            // s64,s32 -> s16
	TimeMinute,          // s64,s32 -> s16
	TimeSecond,          // s64,s32 -> s16
	TimeMillisecond      // s64     -> s16
};

static std::string DateOperationString(DateOperation dateOp)
{
	switch (dateOp)
	{
		case DateOperation::Date:
			return "date";
		case DateOperation::DateYear:
			return "date_year";
		case DateOperation::DateMonth:
			return "date_month";
		case DateOperation::DateDay:
			return "date_day";
		case DateOperation::Time:
			return "time";
		case DateOperation::TimeHour:
			return "time_hour";
		case DateOperation::TimeMinute:
			return "time_minute";
		case DateOperation::TimeSecond:
			return "time_second";
		case DateOperation::TimeMillisecond:
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
class DateGenerator<B, PTX::Int32Type> : public BuiltinGenerator<B, PTX::Int32Type>
{
public:
	DateGenerator(Builder& builder, DateOperation dateOp) : BuiltinGenerator<B, PTX::Int32Type>(builder), m_dateOp(dateOp) {}

	std::string Name() const override { return "DateGenerator"; }

	PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<const HorseIR::Operand *>& arguments) override
	{
		return OperandCompressionGenerator::UnaryCompressionRegister(this->m_builder, arguments);
	}

	PTX::Register<PTX::Int32Type> *Generate(const HorseIR::LValue *target, const std::vector<const HorseIR::Operand *>& arguments) override
	{
		if (m_dateOp != DateOperation::Date)
		{
			BuiltinGenerator<B, PTX::Int32Type>::Unimplemented("date operation " + DateOperationString(m_dateOp));
		}

		OperandGenerator<B, PTX::Int64Type> opGen(this->m_builder);
		auto src = opGen.GenerateOperand(arguments.at(0), OperandGenerator<B, PTX::Int64Type>::LoadKind::Vector);

		auto targetRegister = this->GenerateTargetRegister(target, arguments);

		// date = (dt / 1000) % SECONDS_PER_DAY

		auto resources = this->m_builder.GetLocalResources();
		auto temp1 = resources->template AllocateTemporary<PTX::Int64Type>();
		auto temp2 = resources->template AllocateTemporary<PTX::Int64Type>();
		auto temp3 = resources->template AllocateTemporary<PTX::Int64Type>();

		this->m_builder.AddStatement(new PTX::DivideInstruction<PTX::Int64Type>(temp1, src, new PTX::Int64Value(1000)));
		this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::Int64Type>(temp2, temp1, new PTX::Int64Value(Utils::Date::SECONDS_PER_DAY)));
		this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::Int64Type>(temp3, temp1, temp2));

		ConversionGenerator::ConvertSource<PTX::Int32Type>(this->m_builder, targetRegister, temp3);

		return targetRegister;
	}

private:
	DateOperation m_dateOp;
};

template<PTX::Bits B>
class DateGenerator<B, PTX::Int64Type> : public BuiltinGenerator<B, PTX::Int64Type>
{
public:
	DateGenerator(Builder& builder, DateOperation dateOp) : BuiltinGenerator<B, PTX::Int64Type>(builder), m_dateOp(dateOp) {}

	std::string Name() const override { return "DateGenerator"; }

	PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<const HorseIR::Operand *>& arguments) override
	{
		return OperandCompressionGenerator::UnaryCompressionRegister(this->m_builder, arguments);
	}

	PTX::Register<PTX::Int64Type> *Generate(const HorseIR::LValue *target, const std::vector<const HorseIR::Operand *>& arguments) override
	{
		if (m_dateOp != DateOperation::Time)
		{
			BuiltinGenerator<B, PTX::Int64Type>::Unimplemented("date operation " + DateOperationString(m_dateOp));
		}

		OperandGenerator<B, PTX::Int64Type> opGen(this->m_builder);
		auto src = opGen.GenerateOperand(arguments.at(0), OperandGenerator<B, PTX::Int64Type>::LoadKind::Vector);

		auto targetRegister = this->GenerateTargetRegister(target, arguments);
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(targetRegister, src));

		return targetRegister;
	}

private:
	DateOperation m_dateOp;

};

template<PTX::Bits B>
class DateGenerator<B, PTX::Int16Type> : public BuiltinGenerator<B, PTX::Int16Type>
{
public:
	DateGenerator(Builder& builder, DateOperation dateOp) : BuiltinGenerator<B, PTX::Int16Type>(builder), m_dateOp(dateOp) {}

	std::string Name() const override { return "DateGenerator"; }

	PTX::Register<PTX::PredicateType> *GenerateCompressionPredicate(const std::vector<const HorseIR::Operand *>& arguments) override
	{
		return OperandCompressionGenerator::UnaryCompressionRegister(this->m_builder, arguments);
	}

	PTX::Register<PTX::Int16Type> *Generate(const HorseIR::LValue *target, const std::vector<const HorseIR::Operand *>& arguments) override
	{
		auto targetRegister = this->GenerateTargetRegister(target, arguments);

		if (m_dateOp == DateOperation::TimeMillisecond)
		{
			OperandGenerator<B, PTX::Int64Type> opGen(this->m_builder);
			auto src = opGen.GenerateOperand(arguments.at(0), OperandGenerator<B, PTX::Int64Type>::LoadKind::Vector);
			
			// millisecond = time % 1000

			auto resources = this->m_builder.GetLocalResources();
			auto temp = resources->template AllocateTemporary<PTX::Int64Type>();

			this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::Int64Type>(temp, src, new PTX::Int64Value(1000)));
			ConversionGenerator::ConvertSource<PTX::Int16Type, PTX::Int64Type>(this->m_builder, targetRegister, temp);
		}
		else
		{
			DispatchType(*this, arguments.at(0)->GetType(), targetRegister, arguments.at(0));
		}

		return targetRegister;
	}
	
	template<class T>
	void GenerateVector(PTX::Register<PTX::Int16Type> *target, const HorseIR::Operand *argument)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Convert extended format, truncating milliseconds by /1000

		OperandGenerator<B, T> opGen(this->m_builder);
		auto inputSrc = opGen.GenerateOperand(argument, OperandGenerator<B, T>::LoadKind::Vector);

		PTX::TypedOperand<PTX::Int32Type> *src = nullptr;
		if constexpr(std::is_same<T, PTX::Int32Type>::value)
		{
			src = inputSrc;
		}
		else if constexpr(std::is_same<T, PTX::Int64Type>::value)
		{
			auto temp = resources->template AllocateTemporary<PTX::Int64Type>();
			this->m_builder.AddStatement(new PTX::DivideInstruction<PTX::Int64Type>(temp, inputSrc, new PTX::Int64Value(1000)));
			src = ConversionGenerator::ConvertSource<PTX::Int32Type, PTX::Int64Type>(this->m_builder, temp);
		}
		else
		{
			BuiltinGenerator<B, PTX::Int16Type>::Unimplemented("type for date operation " + DateOperationString(m_dateOp));
		}

		switch (m_dateOp)
		{
			case DateOperation::DateYear:
			{
				GenerateYear(target, src);
				break;
			}
			case DateOperation::DateMonth:
			{
				GenerateMonth(target, src);
				break;
			}
			case DateOperation::DateDay:
			{
				GenerateDay(target, src);
				break;
			}
			case DateOperation::TimeHour:
			{
				// hour = (time % SECONDS_PER_DAY) / SECONDS_PER_HOUR

				auto resources = this->m_builder.GetLocalResources();
				auto temp1 = resources->template AllocateTemporary<PTX::Int32Type>();
				auto temp2 = resources->template AllocateTemporary<PTX::Int32Type>();

				this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::Int32Type>(temp1, src, new PTX::Int32Value(Utils::Date::SECONDS_PER_DAY)));
				this->m_builder.AddStatement(new PTX::DivideInstruction<PTX::Int32Type>(temp2, temp1, new PTX::Int32Value(Utils::Date::SECONDS_PER_HOUR)));
				ConversionGenerator::ConvertSource<PTX::Int16Type, PTX::Int32Type>(this->m_builder, target, temp2);
				break;
			}
			case DateOperation::TimeMinute:
			{
				// minute = (time % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE

				auto resources = this->m_builder.GetLocalResources();
				auto temp1 = resources->template AllocateTemporary<PTX::Int32Type>();
				auto temp2 = resources->template AllocateTemporary<PTX::Int32Type>();

				this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::Int32Type>(temp1, src, new PTX::Int32Value(Utils::Date::SECONDS_PER_HOUR)));
				this->m_builder.AddStatement(new PTX::DivideInstruction<PTX::Int32Type>(temp2, temp1, new PTX::Int32Value(Utils::Date::SECONDS_PER_MINUTE)));
				ConversionGenerator::ConvertSource<PTX::Int16Type, PTX::Int32Type>(this->m_builder, target, temp2);
				break;
			}
			case DateOperation::TimeSecond:
			{
				// second = time % SECONDS_PER_MINUTE

				auto resources = this->m_builder.GetLocalResources();
				auto temp = resources->template AllocateTemporary<PTX::Int32Type>();

				this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::Int32Type>(temp, src, new PTX::Int32Value(Utils::Date::SECONDS_PER_MINUTE)));
				ConversionGenerator::ConvertSource<PTX::Int16Type, PTX::Int32Type>(this->m_builder, target, temp);
				break;
			}
			default:
			{
				BuiltinGenerator<B, PTX::Int16Type>::Unimplemented("date operation " + DateOperationString(m_dateOp));
			}
		}
	}

	template<class T>
	void GenerateList(PTX::Register<PTX::Int16Type> *target, const HorseIR::Operand *argument)
	{
		if (this->m_builder.GetInputOptions().IsVectorGeometry())
		{
			BuiltinGenerator<B, PTX::Int16Type>::Unimplemented("list-in-vector");
		}

		// Lists are handled by the vector code through a projection

		GenerateVector<T>(target, argument);
	}

	template<class T>
	void GenerateTuple(unsigned int index, PTX::Register<PTX::Int16Type> *target, const HorseIR::Operand *argument)
	{
		BuiltinGenerator<B, PTX::Int16Type>::Unimplemented("list-in-vector");
	}

	std::tuple<PTX::Register<PTX::Int32Type> *, PTX::Register<PTX::PredicateType> *> GenerateYear(
		PTX::Register<PTX::Int16Type> *year, PTX::TypedOperand<PTX::Int32Type> *src
	) {
		// Extract year from unix time (https://github.com/eblot/newlib/blob/master/newlib/libc/time/mktm_r.c)
		//
		// days = time / SECONDS_PER_DAY
		// year = UNIX_BASE_YEAR
		//
		// while (true) {
		//     year_days = leap(year) ? DAYS_PER_LYEAR : DAYS_PER_YEAR
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

		this->m_builder.AddStatement(new PTX::DivideInstruction<PTX::Int32Type>(days, src, new PTX::Int32Value(Utils::Date::SECONDS_PER_DAY)));
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int16Type>(year, new PTX::Int16Value(Utils::Date::UNIX_BASE_YEAR)));

		auto leapPredicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddInfiniteLoop("DATE", [&](Builder::LoopContext& loopContext)
		{
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

			this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(tempPredicate, leapPredicate_4, leapPredicate_100));
			this->m_builder.AddStatement(new PTX::OrInstruction<PTX::PredicateType>(leapPredicate, tempPredicate, leapPredicate_400));

			auto yearDays = resources->template AllocateTemporary<PTX::Int32Type>();
			this->m_builder.AddStatement(new PTX::SelectInstruction<PTX::Int32Type>(
				yearDays, new PTX::Int32Value(Utils::Date::DAYS_PER_LYEAR), new PTX::Int32Value(Utils::Date::DAYS_PER_YEAR), leapPredicate
			));

			auto endPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::Int32Type>(
				endPredicate, days, yearDays, PTX::Int32Type::ComparisonOperator::Less
			));
			this->m_builder.AddBreakStatement(loopContext, endPredicate);

			this->m_builder.AddStatement(new PTX::AddInstruction<PTX::Int16Type>(year, year, new PTX::Int16Value(1)));
			this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::Int32Type>(days, days, yearDays));
		});

		return {days, leapPredicate};
	}

	PTX::Register<PTX::Int32Type> *GenerateMonth(PTX::Register<PTX::Int16Type> *month, PTX::TypedOperand<PTX::Int32Type> *src)
	{
		// Extract month from unix time (https://github.com/eblot/newlib/blob/master/newlib/libc/time/mktm_r.c)
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

		this->m_builder.AddInfiniteLoop("DATE", [&](Builder::LoopContext& loopContext)
		{
			auto febPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
			auto febLeapPredicate = resources->template AllocateTemporary<PTX::PredicateType>();

			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::Int16Type>(
				febPredicate, month, new PTX::Int16Value(1), PTX::Int16Type::ComparisonOperator::Equal
			));
			this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(febLeapPredicate, febPredicate, leapPredicate));

			// Get month days from constant space

			std::vector<std::int32_t> mdays({31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31});;

			auto moduleResources = this->m_builder.GetGlobalResources();
			auto c_months = new PTX::ArrayVariableAdapter<PTX::Int32Type, Utils::Date::MONTHS_PER_YEAR, PTX::ConstSpace>(
				moduleResources->template AllocateConstVariable<PTX::ArrayType<PTX::Int32Type, Utils::Date::MONTHS_PER_YEAR>>("mdays", mdays)
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
			this->m_builder.AddBreakStatement(loopContext, endPredicate);

			this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::Int32Type>(days, days, monthDays));
		});

		return days;
	}

	void GenerateDay(PTX::Register<PTX::Int16Type> *day, PTX::TypedOperand<PTX::Int32Type> *src)
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
}
