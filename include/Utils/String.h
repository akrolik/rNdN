#pragma once

#include <string>

namespace Utils {

class String
{
public:
	static bool Like(const std::string& needle, const std::string& pattern);
	static std::string ReplaceString(std::string string, const std::string& find, const std::string& replace, bool recursive = false);

private:
	static bool Like_Internal(const char *needleData, const char *patternData, size_t needleSize, size_t patternSize, unsigned int i, unsigned int j);
};

}
