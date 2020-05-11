#pragma once

#include <ctime>
#include <tuple>

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
		time.tm_mon = month - 1;
		time.tm_year = year - 1900;

		time.tm_isdst = 0;
		return std::mktime(&time);
	}

	static std::tuple<std::uint16_t, std::uint8_t, std::uint8_t> DateFromEpoch(std::int32_t epoch)
	{
		auto time = static_cast<std::time_t>(epoch);
		auto tm = std::gmtime(&time);
		return {tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday};
	}

	static std::tuple<std::uint8_t, std::uint8_t, std::uint8_t> TimeFromEpoch(std::int32_t epoch)
	{
		auto time = static_cast<std::time_t>(epoch);
		auto tm = std::gmtime(&time);
		return {tm->tm_hour, tm->tm_min, tm->tm_sec};
	}

	static std::tuple<
		std::uint16_t, std::uint8_t, std::uint8_t,
		std::uint8_t, std::uint8_t, std::uint8_t, std::uint16_t
	> DatetimeFromEpoch(double extendedEpoch)
	{
		auto epoch = static_cast<std::time_t>(extendedEpoch);
		auto millisecond = static_cast<std::uint16_t>((extendedEpoch - epoch) * 1000);

		auto tm = std::gmtime(&epoch);
		return {
			tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
			tm->tm_hour, tm->tm_min, tm->tm_sec, millisecond
		};
	}
};

}
