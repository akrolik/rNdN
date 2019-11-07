#pragma once

#include "HorseIR/Tree/Expressions/Literals/CalendarValue.h"

#include <cstdint>
#include <ostream>
#include <iomanip>

#include "Utils/Date.h"

namespace HorseIR {

class MonthValue : public CalendarValue
{
public:
	MonthValue(std::uint16_t year, std::uint8_t month) : m_year(year), m_month(month) {}

	MonthValue *Clone() const
	{
		return new MonthValue(m_year, m_month);
	}

	std::uint16_t GetYear() const { return m_year; }
	void SetYear(std::uint16_t year) { m_year = year; }

	std::uint8_t GetMonth() const { return m_month; }
	void SetMonth(std::uint8_t month) { m_month = month; }

	std::int32_t GetEpochTime() const override
	{
		return Utils::Date::EpochTime_day(m_year, m_month);
	}

	friend std::ostream& operator<<(std::ostream& os, const MonthValue& value);

	bool operator==(const MonthValue& other) const
	{
		return (m_year == other.m_year && m_month == other.m_month);
	}

	bool operator!=(const MonthValue& other) const
	{
		return !(*this == other);
	}
	
protected:
	std::uint16_t m_year = 0;
	std::uint8_t m_month = 0;
};

inline std::ostream& operator<<(std::ostream& os, const MonthValue& value)
{
	os << std::setfill('0') << std::setw(4) << value.m_year;
	os << "-";
	os << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(value.m_month);
	return os;
}

}
