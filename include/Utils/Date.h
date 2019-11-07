#pragma once

#include <ctime>

namespace Utils {

class Date
{
public:
	static double ExtendedEpochTime_time(std::uint8_t hour, std::uint8_t minute, std::uint8_t second, std::uint16_t millisecond)
	{
		auto epoch = EpochTime_time(hour, minute, second);
		auto extend = static_cast<double>(millisecond) / 1000;
		return epoch + extend;
	}

	static double ExtendedEpochTime(
		std::uint16_t year, std::uint8_t month, std::uint8_t day,
		std::uint8_t hour, std::uint8_t minute, std::uint8_t second, std::uint16_t millisecond
	)
	{
		auto epoch = EpochTime(year, month, day, hour, minute, second);
		auto extend = static_cast<double>(millisecond) / 1000;
		return epoch + extend;
	}

	static std::int32_t EpochTime_day(std::uint16_t year, std::uint8_t month, std::uint8_t day = 1)
	{
		return EpochTime(year, month, day, 0, 0, 0);
	}

	static std::int32_t EpochTime_time(std::uint8_t hour, std::uint8_t minute, std::uint8_t second = 0)
	{
		return EpochTime(1900, 1, 1, hour, minute, second);
	}

	static std::int32_t EpochTime(
		std::uint16_t year, std::uint8_t month, std::uint8_t day,
		std::uint8_t hour, std::uint8_t minute, std::uint8_t second
	)
	{
		struct std::tm time;
		time.tm_sec = second;
		time.tm_min = minute;
		time.tm_hour = hour;

		time.tm_mday = day;
		time.tm_mon= month - 1;
		time.tm_year = year - 1900;

		time.tm_isdst = 0;
		return std::mktime(&time);
	}
};

}
