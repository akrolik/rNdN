#pragma once

#include "HorseIR/Tree/Expressions/Literals/ExtendedCalendarValue.h"

#include <ostream>

#include "HorseIR/Tree/Expressions/Literals/DateValue.h"
#include "HorseIR/Tree/Expressions/Literals/TimeValue.h"

#include "Utils/Date.h"

namespace HorseIR {

class DatetimeValue : public ExtendedCalendarValue
{
public:
	DatetimeValue(DateValue *date, TimeValue *time) : m_date(date), m_time(time) {}

	DatetimeValue *Clone() const
	{
		return new DatetimeValue(m_date->Clone(), m_time->Clone());
	}

	// Date

	const DateValue *GetDate() const { return m_date; }
	DateValue *GetDate() { return m_date; }
	void SetDate(DateValue *date) { m_date = date; }

	// Time

	const TimeValue *GetTime() const { return m_time; }
	TimeValue *GetTime() { return m_time; }
	void SetTime(TimeValue *time) { m_time = time; }

	// Formatting

	std::int64_t GetExtendedEpochTime() const override
	{
		return Utils::Date::ExtendedEpochTime(
			m_date->GetYear(), m_date->GetMonth(), m_date->GetDay(),
			m_time->GetHour(), m_time->GetMinute(), m_time->GetSecond(), m_time->GetMillisecond()
		);
	}

	std::string ToString() const
	{
		std::stringstream stream;
		stream << *this;
		return stream.str();
	}

	friend std::ostream& operator<<(std::ostream& os, const DatetimeValue& value);

	// Operators

	bool operator==(const DatetimeValue& other) const
	{
		return (*m_date == *other.m_date && *m_time == *other.m_time);
	}

	bool operator!=(const DatetimeValue& other) const
	{
		return !(*this == other);
	}

protected:
	DateValue *m_date = nullptr;
	TimeValue *m_time = nullptr;
};

inline std::ostream& operator<<(std::ostream& os, const DatetimeValue& value)
{
      os << *value.m_date << "T" << *value.m_time;
      return os;
}

}
