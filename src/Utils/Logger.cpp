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

void Logger::LogInfo(const std::string& info, const std::string& prefix)
{
	if (prefix != NoPrefix)
	{
		std::cout << "[" << prefix << "] ";
	}
	std::cout << info << std::endl;
}

void Logger::LogError(const std::string& error, const std::string& prefix, bool exit)
{
	if (prefix != NoPrefix)
	{
		std::cerr << "[" << prefix << "] ";
	}
	std::cerr << error << std::endl;
	if (exit)
	{
		std::exit(EXIT_FAILURE);
	}
}

void Logger::LogTiming(const std::string& name, long timing)
{
	std::cout << "[TIME] " << name << ": " << time << " mus" << std::endl;
}

void Logger::LogTimingComponent(const std::string& name, long time)
{
	std::cout << "[TIME]  - " << name << ": " << time << " mus" << std::endl;
}

}
