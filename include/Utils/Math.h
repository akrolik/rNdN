#pragma once

#include <cmath>
#include <numeric>

namespace Utils {

class Math
{
public:

template<typename T>
static T Power2(T value)
{
	return static_cast<T>(std::pow(2, std::ceil(std::log2(value))));
}

};

}
