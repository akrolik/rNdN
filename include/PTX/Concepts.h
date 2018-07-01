#pragma once

#include "PTX/Utils.h"

namespace PTX {

template<template<class, class> class C, class T, class... R>
struct Enforcer
{
	constexpr static bool value = is_one<C<R, T>::value...>::value;
};

constexpr static bool Assert = true;

#define REQUIRE_EXACT(P, ...) \
	Enforcer<std::is_same, P, __VA_ARGS__>::value

#define REQUIRE_BASE(P, ...) \
	Enforcer<std::is_base_of, P, __VA_ARGS__>::value

}
