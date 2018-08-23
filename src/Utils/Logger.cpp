#include "Utils/Logger.h"

#include <iostream>

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
		_prefix += std::string(indentation, ' ') + "- ";
	}
	std::cout << _prefix << ReplaceString(info, "\n", "\n" + _prefix);
	if (newline)
	{
		std::cout << std::endl;
	}
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

void Logger::LogTiming(const std::string& name, long time)
{
	std::cout << "[TIME] " << name << ": " << time << " mus" << std::endl;
}

void Logger::LogTimingComponent(const std::string& name, long time)
{
	std::cout << "[TIME]  - " << name << ": " << time << " mus" << std::endl;
}

void Logger::LogBlank(const std::string& prefix)
{
	std::cout << prefix << std::endl;
}

}
