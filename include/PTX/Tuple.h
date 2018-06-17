#pragma once

namespace PTX {

//TODO: comment
template<std::size_t> struct int_{};

template <typename T, size_t P>
static std::ostringstream& CodeTuple(std::ostringstream& code, std::string separator, const T& t, int_<P>)
{
	auto arg = std::get<std::tuple_size<T>::value-P>(t);
	if (arg == nullptr)
	{
		std::cerr << "[ERROR] Parameter " << std::tuple_size<T>::value-P << " not set" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	code << arg->ToString();
	if (P > 1)
	{
		code << ",";
	}
	code << separator;
	return CodeTuple(code, separator, t, int_<P-1>());
}

template <typename T>
static std::ostringstream& CodeTuple(std::ostringstream& code, std::string separator, const T& t, int_<0>)
{
	return code;
}

}