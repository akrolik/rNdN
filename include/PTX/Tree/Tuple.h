#pragma once

#include <sstream>
#include <tuple>

#include "Utils/Logger.h"

namespace PTX {

template<std::size_t> struct int_{};

template <typename V, typename T, size_t P>
static void ExpandTuple(std::vector<V *>& vector, T& t, int_<P>)
{
	auto arg = std::get<std::tuple_size<T>::value-P>(t);
	if (arg == nullptr)
	{
		Utils::Logger::LogError("Element " + std::to_string(std::tuple_size<T>::value-P) + " not set");
	}
	vector.push_back(arg);
	ExpandTuple(vector, t, int_<P-1>());
}

template <typename V, typename T>
static void ExpandTuple(std::vector<V *>& operands, T& t, int_<0>) {}

}
