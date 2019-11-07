#pragma once

#include "HorseIR/Tree/Expressions/Literals/CalendarValue.h"

#include <cstdint>
#include <ostream>
#include <iomanip>

#include "Utils/Date.h"

namespace HorseIR {

class SecondValue : public CalendarValue
{
public:
	SecondValue(std::uint8_t hour, std::uint8_t minute, std::uint8_t second) : m_hour(hour), m_minute(minute), m_second(second) {}

	SecondValue *Clone() const
	{
		return new SecondValue(m_hour, m_minute, m_second);
	}

	std::uint8_t GetHour() const { return m_hour; }
	void SetHour(std::uint8_t hour) { m_hour = hour; }

	std::uint8_t GetMinute() const { return m_minute; }
	void SetMinute(std::uint8_t minute) { m_minute = minute; }

	std::uint8_t GetSecond() const { return m_second; }
	void SetSecond(std::uint8_t second) { m_second = second; }

	std::int32_t GetEpochTime() const override
	{
		return Utils::Date::EpochTime_time(m_hour, m_minute, m_second);
	}

	friend std::ostream& operator<<(std::ostream& os, const SecondValue& value);

	bool operator==(const SecondValue& other) const
	{
		return (m_hour == other.m_hour &&
			m_minute == other.m_minute &&
			m_second == other.m_second);
	}

	bool operator!=(const SecondValue& other) const
	{
		return !(*this == other);
	}
	
protected:
	std::uint8_t m_hour = 0;
	std::uint8_t m_minute = 0;
	std::uint8_t m_second = 0;
};

inline std::ostream& operator<<(std::ostream& os, const SecondValue& value)
{
	os << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(value.m_hour);
	os << ":";
	os << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(value.m_minute);
	os << ":";
	os << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(value.m_second);
	return os;
}

}
