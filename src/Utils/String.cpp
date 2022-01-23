#include "Utils/String.h"

#include "Utils/Logger.h"

namespace Utils {

std::string String::ReplaceString(std::string string, const std::string& needle, const std::string& replace, bool recursive)
{
	// Inspired by: https://stackoverflow.com/questions/1494399/how-do-i-search-find-and-replace-in-a-standard-string

	auto position = 0;
	auto size = needle.size();

	auto found = string.find(needle, position);
	while (found != std::string::npos)
	{
		// Replace the current occurence

		string.replace(found, size, replace);

		// Find the next occurence after the current replacement if not recursive

		if (!recursive)
		{
			position = found + size;
		}
		found = string.find(needle, position);
	}

	return string;
}

bool String::Like(const std::string& needle, const std::string& pattern)
{
	auto needleSize = needle.size();
	auto patternSize = pattern.size();

	if (needleSize == 0 || patternSize == 0)
	{
		return false;
	}

	auto needleData = needle.c_str();
	auto patternData = pattern.c_str();

	return Like(needleData, patternData);
}

bool String::Like(const char *needleData, const char *patternData)
{
	// Inspired by: https://www.codeproject.com/Tips/608266/A-Csharp-LIKE-implementation-that-mimics-SQL-LIKE

	while (true)
	{
		auto pc = *patternData;
		if (pc == '\0')
		{
			break;
		}
		else if (pc == '\\' || pc == '_' || pc == '[')
		{
			Utils::Logger::LogError("Unsupport pattern character '" + std::string(pc, 1) + "'");
		}
		else if (pc == '%')
		{
			auto pc1 = *(patternData + 1);
			if (pc1 == '\0')
			{
				return true;
			}

			while (true)
			{
				auto nc = *needleData;
				if (nc == '\0')
				{
					return false;
				}
				else if (nc == pc1)
				{
					if (Like(needleData + 1, patternData + 2))
					{
						return true;
					}
				}
				++needleData;
			}
		}
		else
		{
			auto pj = *needleData;
			if (pj == '\0')
			{
				return false;
			}
			else if (pj == pc)
			{
				++needleData;
			}
			else
			{
				return false;
			}
		}
		++patternData;
	}

	return (*needleData == '\0');
}

}
