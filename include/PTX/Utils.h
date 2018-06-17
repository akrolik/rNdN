#pragma once

namespace PTX {

//TODO: comment
template <bool... B> struct is_one;

template <bool... T>
struct is_one<true, T...> : std::true_type {};

template <bool... T>
struct is_one<false, T...> : is_one<T...> {};

template <> struct is_one<> : std::false_type {};

//TODO: comment
template <bool... B> struct is_all;

template <bool... T>
struct is_all<true, T...> : is_all<T...> {};

template <bool... T>
struct is_all<false, T...> : std::false_type {};

template <> struct is_all<> : std::true_type {};

}
