#include "Utils/Logger.h"

#include <iostream>
#include <iomanip>

#include "Utils/Options.h"

namespace Utils {

void Logger::LogSection(const std::string& name, bool separate)
{
	if (separate)
	{
		std::cout << std::endl;
	}
	std::cout << name << std::endl;
}

static std::string ReplaceString(std::string subject, const std::string& search, const std::string& replace)
{
	size_t pos = 0;
	while ((pos = subject.find(search, pos)) != std::string::npos)
	{
		subject.replace(pos, search.length(), replace);
		pos += replace.length();
	}
	return subject;
}

void Logger::LogInfo(const std::string& info, unsigned int indentation, bool newline, const std::string& prefix)
{
	std::string _prefix = "";
	if (prefix != NoPrefix)
	{
		_prefix = "[" + prefix + "] ";
	}
	if (indentation)
	{
		_prefix += std::string(indentation * 2, ' ') + "- ";
	}
	std::cout << _prefix << ReplaceString(info, "\n", "\n" + _prefix);
	if (newline)
	{
		std::cout << std::endl;
	}
}

void Logger::LogDebug(const std::string& info, unsigned int indentation, bool newline)
{
	LogInfo(info, indentation, newline, DebugPrefix);
}

void Logger::LogError(const std::string& error, const std::string& prefix)
{
	LogErrorPart(error, prefix);
	std::exit(EXIT_FAILURE);
}

void Logger::LogErrorPart(const std::string& error, const std::string& prefix)
{
	std::string _prefix = "";
	if (prefix != NoPrefix)
	{
		_prefix = "[" + prefix + "] ";
	}
	std::cerr << _prefix << ReplaceString(error, "\n", "\n" + _prefix) << std::endl;
}

void Logger::LogTiming(const Chrono::Timing *timing, unsigned int indentation)
{
	auto ratio = 1;
	auto unitString = "ns";

	if (Utils::Options::GetDebug_TimeUnit() == Utils::Options::TimeUnit::Microseconds)
	{
		ratio = 1000;
		unitString = "us";
	}
	
	auto name = timing->GetName();
	auto time = double(timing->GetTime()) / ratio;

	std::string _prefix = "[TIME] ";
	if (indentation > 0)
	{
		_prefix += std::string(indentation * 2, ' ');
	}

	if (timing->HasChildren())
	{
		std::cout << _prefix << name << std::endl;

		auto childTime = 0.0;
		for (auto child : timing->GetChildren())
		{
			childTime += double(child->GetTime()) / ratio;
			LogTiming(child, indentation + 1);
		}

		auto overhead = (time - childTime);

		std::cout << std::fixed << std::setprecision(1) << _prefix << name << ": " << time << " " << unitString << " [overhead: " << overhead << " " << unitString << "]" << std::endl;
	}
	else
	{
		std::cout << _prefix << name << ": " << time << " " << unitString << std::endl;
	}
}

void Logger::LogBlank(const std::string& prefix)
{
	std::cout << prefix << std::endl;
}

}
