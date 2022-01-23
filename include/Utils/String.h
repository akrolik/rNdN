#pragma once

#include <string>

namespace Utils {

class String
{
public:
	static bool Like(const std::string& needle, const std::string& pattern);
	static bool Like(const char *needleData, const char *patternData);

	static std::string ReplaceString(std::string string, const std::string& find, const std::string& replace, bool recursive = false);
};

}
