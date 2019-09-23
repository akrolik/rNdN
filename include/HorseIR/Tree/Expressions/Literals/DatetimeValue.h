#pragma once

#include <ostream>

#include "HorseIR/Tree/Expressions/Literals/DateValue.h"
#include "HorseIR/Tree/Expressions/Literals/TimeValue.h"

namespace HorseIR {

class DatetimeValue
{
public:
	DatetimeValue(DateValue *date, TimeValue *time) : m_date(date), m_time(time) {}

	DatetimeValue *Clone() const
	{
		return new DatetimeValue(m_date->Clone(), m_time->Clone());
	}

	DateValue *GetDate() const { return m_date; }
	void SetDate(DateValue *date) { m_date = date; }

	TimeValue *GetTime() const { return m_time; }
	void SetTime(TimeValue *time) { m_time = time; }

	friend std::ostream& operator<<(std::ostream& os, const DatetimeValue& value);

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
      os << value.m_date << "T" << value.m_time;
      return os;
}

}
