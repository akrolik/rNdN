#pragma once

#include <string>

namespace Utils {

class String
{
public:
	static bool Like(const std::string& needle, const std::string& pattern);

private:
	static bool Like_Internal(const char *needleData, const char *patternData, size_t needleSize, size_t patternSize, unsigned int i, unsigned int j);
};

}
