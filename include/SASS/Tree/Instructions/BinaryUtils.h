#pragma once

namespace SASS {

#define SASS_ENUM_FRIEND(x) \
	friend bool operator!(x a); \
	friend x operator|(x a, x b); \
	friend x operator&(x a, x b); \
	friend x& operator|=(x& a, x b); \
	friend x& operator&=(x& a, x b);

#define SASS_ENUM_INLINE(x,y) \
	inline bool operator!(x::y a) \
	{ \
		return static_cast<std::underlying_type_t<x::y>>(a) == 0; \
	} \
	inline x::y operator&(x::y a, x::y b) \
	{ \
		return static_cast<x::y>( \
			static_cast<std::underlying_type_t<x::y>>(a) & \
			static_cast<std::underlying_type_t<x::y>>(b) \
		); \
	} \
	inline x::y& operator&=(x::y& a, x::y b) \
	{ \
		    return a = a & b; \
	} \
	inline x::y operator|(x::y a, x::y b) \
	{ \
		return static_cast<x::y>( \
			static_cast<std::underlying_type_t<x::y>>(a) | \
			static_cast<std::underlying_type_t<x::y>>(b) \
		); \
	} \
	inline x::y& operator|=(x::y& a, x::y b) \
	{ \
		    return a = a | b; \
	}

#define SASS_FLAGS_FRIEND() SASS_ENUM_FRIEND(Flags)
#define SASS_FLAGS_INLINE(x) SASS_ENUM_INLINE(x, Flags)

class BinaryUtils
{
public:
	static std::uint64_t Format(std::uint64_t value, std::uint8_t shift, std::uint64_t mask)
	{
		return (value & mask) << shift;
	}

	template <typename E>
	static std::uint64_t Format(E value, std::uint8_t shift, std::uint64_t mask)
	{
		    return Format(static_cast<std::uint64_t>(value), shift, mask);
	}
};

}
