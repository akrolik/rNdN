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

	return Like_Internal(needleData, patternData, needleSize, patternSize, 0, 0);
}

bool String::Like_Internal(const char *needleData, const char *patternData, size_t needleSize, size_t patternSize, unsigned int i, unsigned int j)
{
	// Inspired by: https://www.codeproject.com/Tips/608266/A-Csharp-LIKE-implementation-that-mimics-SQL-LIKE

	for (;i < patternSize; ++i)
	{
		auto pc = patternData[i];
		if (pc == '\\' || pc == '_' || pc == '[')
		{
			Utils::Logger::LogError("Unsupport pattern character '" + std::string(pc, 1) + "'");
		}
		else if (pc == '%')
		{
			if (i + 1 == patternSize)
			{
				return true;
			}

			while (j < needleSize)
			{
				if (needleData[j] == patternData[i + 1])
				{
					if (String::Like_Internal(needleData, patternData, needleSize, patternSize, i + 2, j + 1))
					{
						return true;
					}
				}
				++j;
			}

			return false;
		}
		else if (j >= needleSize)
		{
			return false;
		}
		else if (needleData[j] == patternData[i])
		{
			++j;
		}
		else
		{
			return false;
		}
	}

	return (j == needleSize);
}

}
