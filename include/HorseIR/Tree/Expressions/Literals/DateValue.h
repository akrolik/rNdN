#pragma once

#include <cstdint>
#include <ctime>
#include <ostream>
#include <iomanip>

namespace HorseIR {

class DateValue
{
public:
	DateValue(std::uint16_t year, std::uint8_t month, std::uint8_t day) : m_year(year), m_month(month), m_day(day) {}

	DateValue *Clone() const
	{
		return new DateValue(m_year, m_month, m_day);
	}

	std::uint16_t GetYear() const { return m_year; }
	void SetYear(std::uint16_t year) { m_year = year; }

	std::uint8_t GetMonth() const { return m_month; }
	void SetMonth(std::uint8_t month) { m_month = month; }

	std::uint8_t GetDay() const { return m_day; }
	void SetDay(std::uint8_t day) { m_day = day; }

	std::int32_t GetEpochTime() const
	{
		struct std::tm time;
		time.tm_sec = 0;
		time.tm_min = 0;
		time.tm_hour = 0;

		time.tm_mday = m_day;
		time.tm_mon= m_month - 1;
		time.tm_year = m_year - 1900;

		time.tm_isdst = 0;
		return std::mktime(&time);
	}

	friend std::ostream& operator<<(std::ostream& os, const DateValue& value);

	bool operator==(const DateValue& other) const
	{
		return (m_year == other.m_year &&
			m_month == other.m_month &&
			m_day == other.m_day);
	}

	bool operator!=(const DateValue& other) const
	{
		return !(*this == other);
	}
	
protected:
	std::uint16_t m_year = 0;
	std::uint8_t m_month = 0;
	std::uint8_t m_day = 0;
};

inline std::ostream& operator<<(std::ostream& os, const DateValue& value)
{
	os << std::setfill('0') << std::setw(4) << value.m_year;
	os << "-";
	os << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(value.m_month);
	os << "-";
	os << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(value.m_day);
	return os;
}

}
