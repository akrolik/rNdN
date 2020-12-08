#pragma once

#include <ctime>
#include <tuple>

namespace Utils {

class Date
{
public:
	constexpr static auto UNIX_BASE_YEAR = 1970;
	constexpr static auto UNIX_YEAR_OFFSET = 1900;

	constexpr static auto MONTHS_PER_YEAR = 12;

	constexpr static auto DAYS_PER_YEAR = 365;
	constexpr static auto DAYS_PER_LYEAR = 366;

	constexpr static auto SECONDS_PER_MINUTE = 60;
	constexpr static auto MINUTES_PER_HOUR = 60;
	constexpr static auto HOURS_PER_DAY = 24;

	constexpr static auto SECONDS_PER_HOUR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR;
	constexpr static auto SECONDS_PER_DAY = SECONDS_PER_HOUR * HOURS_PER_DAY;
	constexpr static auto SECONDS_PER_YEAR = SECONDS_PER_DAY * DAYS_PER_YEAR;

	constexpr static std::uint16_t MDAYS_YEAR[2][12] = {
		{ 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334 },
		{ 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335 }
	};

	static std::int64_t ExtendedEpochTime_time(std::uint8_t hour, std::uint8_t minute, std::uint8_t second, std::uint16_t millisecond)
	{
		auto epoch = static_cast<std::int64_t>(EpochTime_time(hour, minute, second));
		return (epoch * 1000) + millisecond;
	}

	static std::int64_t ExtendedEpochTime(
		std::uint16_t year, std::uint8_t month, std::uint8_t day,
		std::uint8_t hour, std::uint8_t minute, std::uint8_t second, std::uint16_t millisecond
	)
	{
		auto epoch = static_cast<std::int64_t>(EpochTime(year, month, day, hour, minute, second));
		return (epoch * 1000) + millisecond;
	}

	static std::int32_t EpochTime_day(std::uint16_t year, std::uint8_t month, std::uint8_t day = 1)
	{
		return EpochTime(year, month, day, 0, 0, 0);
	}

	static std::int32_t EpochTime_time(std::uint8_t hour, std::uint8_t minute, std::uint8_t second = 0)
	{
		return EpochTime(UNIX_BASE_YEAR, 1, 1, hour, minute, second);
	}

	static bool IsLeapYear(std::uint16_t year)
	{
		return ((year % 4 == 0 && year % 100 != 0) || year % 400 == 0);
	}

	static std::int32_t EpochTime(
		std::uint16_t year, std::uint8_t month, std::uint8_t day,
		std::uint8_t hour, std::uint8_t minute, std::uint8_t second
	)
	{
		auto epoch = 0;

		epoch += second;
		epoch += minute * SECONDS_PER_MINUTE;
		epoch += hour * SECONDS_PER_HOUR;

		epoch += (day - 1) * SECONDS_PER_DAY;
		epoch += MDAYS_YEAR[IsLeapYear(year)][month - 1] * SECONDS_PER_DAY;

		auto oyear = year - UNIX_YEAR_OFFSET;
		epoch += (oyear - 70) * SECONDS_PER_YEAR;
		epoch += ((oyear - 69)/4 - (oyear - 1)/100 + (oyear + 299)/400) * SECONDS_PER_DAY;

		return epoch;

		// struct std::tm time;
		// time.tm_sec = second;
		// time.tm_min = minute;
		// time.tm_hour = hour;

		// time.tm_mday = day;
		// time.tm_mon = month - 1;
		// time.tm_year = year - UNIX_YEAR_OFFSET;

		// return std::mktime(&time);
	}

	static std::tuple<std::uint16_t, std::uint8_t, std::uint8_t> DateFromEpoch(std::int32_t epoch)
	{
		auto time = static_cast<std::time_t>(epoch);
		auto tm = std::gmtime(&time);
		return {tm->tm_year + UNIX_YEAR_OFFSET, tm->tm_mon + 1, tm->tm_mday};
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
	> DatetimeFromEpoch(std::int64_t extendedEpoch)
	{
		auto time = static_cast<std::time_t>(extendedEpoch / 1000);
		auto millisecond = static_cast<std::uint16_t>(extendedEpoch % 1000);

		auto tm = std::gmtime(&time);
		return {
			tm->tm_year + UNIX_YEAR_OFFSET, tm->tm_mon + 1, tm->tm_mday,
			tm->tm_hour, tm->tm_min, tm->tm_sec, millisecond
		};
	}
};

}
