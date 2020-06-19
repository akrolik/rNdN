#pragma once

#include <cmath>
#include <numeric>

namespace Utils {

class Math
{
public:

template<typename T>
static T Log2(T value)
{
	return static_cast<T>(std::log2(value));
}

template<typename T>
static T Power2(T value)
{
	return static_cast<T>(std::pow(2, std::ceil(std::log2(value))));
}

template<template<typename> class V, typename T>
static T Average(const V<T>& values)
{
	return std::accumulate(std::begin(values), std::end(values), 0) / values.size();
}

};

}
