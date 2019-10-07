#pragma once

#include <chrono>

namespace Utils {

class Chrono
{
public:
	using TimeTy = std::chrono::time_point<std::chrono::steady_clock>;

	static TimeTy Start();
	static long End(TimeTy start);

private:
	Chrono() {}
};

}
