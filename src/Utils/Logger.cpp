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

void Logger::LogError(const std::string& error, const std::string& prefix)
{
	if (prefix != NoPrefix)
	{
		std::cerr << "[" << prefix << "] ";
	}
	std::cerr << error << std::endl;
	std::exit(EXIT_FAILURE);
}

void Logger::LogErrorPart(const std::string& error, const std::string& prefix)
{
	if (prefix != NoPrefix)
	{
		std::cerr << "[" << prefix << "] ";
	}
	std::cerr << error << std::endl;
}

void Logger::LogTiming(const std::string& name, long time)
{
	std::cout << "[TIME] " << name << ": " << time << " mus" << std::endl;
}

void Logger::LogTimingComponent(const std::string& name, long time)
{
	std::cout << "[TIME]  - " << name << ": " << time << " mus" << std::endl;
}

}
