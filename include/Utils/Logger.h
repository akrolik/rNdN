#pragma once

#include <iostream>

namespace Utils {

class Logger
{
public:
	constexpr static char const *NoPrefix = "";
	constexpr static char const *InfoPrefix = "INFO";
	constexpr static char const *DebugPrefix = "DEBUG";
	constexpr static char const *ErrorPrefix = "ERROR";

	Logger(Logger const&) = delete;
	void operator=(Logger const&) = delete;

	static void LogSection(const std::string& name, bool separate = true);

	static void LogInfo(const std::string& info, unsigned int indentation = 0, const std::string& prefix = InfoPrefix);

	[[noreturn]] static void LogError(const std::string& error, const std::string& prefix = ErrorPrefix);
	static void LogErrorPart(const std::string& error, const std::string& prefix = ErrorPrefix);

	static void LogTiming(const std::string& name, long time);
	static void LogTimingComponent(const std::string& name, long time);

private:
	Logger() {}

	static Logger& GetInstance()
	{
		static Logger instance;
		return instance;
	}

};

}
