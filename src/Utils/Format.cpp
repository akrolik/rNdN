#include "Utils/Format.h"

#include <sstream>
#include <iomanip>

namespace Utils {

std::string Format::HexString(std::uint64_t value, std::uint8_t fill)
{
	std::ostringstream hex;
	hex << "0x" << std::hex;
	if (fill > 0)
	{
		hex << std::setfill('0') << std::setw(fill);
	}
	hex << value;
	return hex.str();
}

}
