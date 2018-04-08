#pragma once

namespace PTX {

typedef int8_t Int8;
typedef int16_t Int16;
typedef int32_t Int32;
typedef int64_t Int64;

typedef uint8_t UInt8; 
typedef uint16_t UInt16;
typedef uint32_t UInt32;
typedef uint64_t UInt64;

typedef float Float16;
typedef struct { float _1; float _2; } Float16x2;
typedef float Float32;
typedef double Float64;

typedef int8_t Bit8;
typedef int16_t Bit16;
typedef int32_t Bit32;
typedef int64_t Bit64;

typedef int64_t Predicate;

template<typename T>
std::string TypeName()
{
	return "<unknown>";
}

template<>
inline std::string TypeName<UInt64>() { return std::string(".u64"); }

template<>
inline std::string TypeName<UInt32>() { return std::string(".u32"); }

}
