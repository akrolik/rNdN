#pragma once

namespace PTX {

// @struct is_one
//
// Type trait for checking if at least one argument to a variadic template satisfies some condition

template <bool... B> struct is_one;

template <bool... T>
struct is_one<true, T...> : std::true_type {};

template <bool... T>
struct is_one<false, T...> : is_one<T...> {};

template <> struct is_one<> : std::false_type {};

// @struct is_all
//
// Type trait for checking if all argument to a variadic template satisfies some condition

template <bool... B> struct is_all;

template <bool... T>
struct is_all<true, T...> : is_all<T...> {};

template <bool... T>
struct is_all<false, T...> : std::false_type {};

template <> struct is_all<> : std::true_type {};

}
