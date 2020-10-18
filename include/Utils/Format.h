#pragma once

#include <string>

namespace Utils {

class Format
{
public:
	static std::string HexString(std::uint64_t value, std::uint8_t fill = 0);
};

}
