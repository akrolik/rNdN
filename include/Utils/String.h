#pragma once

#include <string>

namespace Utils {

class String
{
public:
	static bool Like(const std::string& needle, const std::string& pattern);
	static std::string ReplaceString(std::string subject, const std::string& search, const std::string& replace);

private:
	static bool Like_Internal(const char *needleData, const char *patternData, size_t needleSize, size_t patternSize, unsigned int i, unsigned int j);
};

}
