#include "Utils/Chrono.h"

#include <iostream>

namespace Utils {

Chrono::TimeTy Chrono::Start()
{
	return std::chrono::steady_clock::now();
}

long Chrono::End(Chrono::TimeTy start)
{
	auto end = std::chrono::steady_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

}
