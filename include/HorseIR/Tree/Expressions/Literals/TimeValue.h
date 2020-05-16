#pragma once

#include "HorseIR/Tree/Expressions/Literals/ExtendedCalendarValue.h"

#include <cstdint>
#include <ostream>
#include <iomanip>

#include "Utils/Date.h"

namespace HorseIR {

class TimeValue : public ExtendedCalendarValue
{
public:
	TimeValue(std::uint8_t hour, std::uint8_t minute, std::uint8_t second, std::uint16_t millisecond) : m_hour(hour), m_minute(minute), m_second(second), m_millisecond(millisecond) {}

	TimeValue *Clone() const
	{
		return new TimeValue(m_hour, m_minute, m_second, m_millisecond);
	}

	std::uint8_t GetHour() const { return m_hour; }
	void SetHour(std::uint8_t hour) { m_hour = hour; }

	std::uint8_t GetMinute() const { return m_minute; }
	void SetMinute(std::uint8_t minute) { m_minute = minute; }

	std::uint8_t GetSecond() const { return m_second; }
	void SetSecond(std::uint8_t second) { m_second = second; }

	std::uint8_t GetMillisecond() const { return m_millisecond; }
	void SetMillisecond(std::uint8_t millisecond) { m_millisecond = millisecond; }

	std::int64_t GetExtendedEpochTime() const override
	{
		return Utils::Date::ExtendedEpochTime_time(m_hour, m_minute, m_second, m_millisecond);
	}

	friend std::ostream& operator<<(std::ostream& os, const TimeValue& value);

	std::string ToString() const
	{
		std::stringstream stream;
		stream << *this;
		return stream.str();
	}

	bool operator==(const TimeValue& other) const
	{
		return (m_hour == other.m_hour &&
			m_minute == other.m_minute &&
			m_second == other.m_second &&
			m_millisecond == other.m_millisecond);
	}

	bool operator!=(const TimeValue& other) const
	{
		return !(*this == other);
	}
	
protected:
	std::uint8_t m_hour = 0;
	std::uint8_t m_minute = 0;
	std::uint8_t m_second = 0;
	std::uint16_t m_millisecond = 0;
};

inline std::ostream& operator<<(std::ostream& os, const TimeValue& value)
{
	os << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(value.m_hour);
	os << ":";
	os << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(value.m_minute);
	os << ":";
	os << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(value.m_second);
	os << ".";
	os << std::setfill('0') << std::setw(3) << value.m_millisecond;
	return os;
}

}
