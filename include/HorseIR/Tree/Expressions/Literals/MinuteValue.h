#pragma once

#include <cstdint>
#include <ostream>
#include <iomanip>

namespace HorseIR {

class MinuteValue
{
public:
	MinuteValue(std::uint8_t hour, std::uint8_t minute) : m_hour(hour), m_minute(minute) {}

	std::uint8_t GetHour() const { return m_hour; }
	void SetHour(std::uint8_t hour) { m_hour = hour; }

	std::uint8_t GetMinute() const { return m_minute; }
	void SetMinute(std::uint8_t minute) { m_minute = minute; }

	friend std::ostream& operator<<(std::ostream& os, const MinuteValue& value);

	bool operator==(const MinuteValue& other) const
	{
		return (m_hour == other.m_hour && m_minute == other.m_minute);
	}

	bool operator!=(const MinuteValue& other) const
	{
		return !(*this == other);
	}
	
protected:
	std::uint8_t m_hour = 0;
	std::uint8_t m_minute = 0;
};

inline std::ostream& operator<<(std::ostream& os, const MinuteValue& value)
{
	os << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(value.m_hour);
	os << ":";
	os << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(value.m_minute);
	return os;
}

}

