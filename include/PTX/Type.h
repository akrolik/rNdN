#pragma once

#include <string>

namespace PTX {

enum VectorSize {
	Scalar,
	Vector2,
	Vector4
};

enum VectorElement {
	X,
	Y,
	Z
};

enum Type {
	Void,

	Int8,
	Int16,
	Int32,
	Int64,

	UInt8,
	UInt16,
	UInt32,
	UInt64,

	Float16,
	Float16x2,
	Float32,
	Float64,

	Bit8,
	Bit16,
	Bit32,
	Bit64,

	Predicate
};

template<Type T> std::string TypeName() { return ".<unknown>"; }

template<> inline std::string TypeName<Int8>() { return std::string(".s8"); }
template<> inline std::string TypeName<Int16>() { return std::string(".s16"); }
template<> inline std::string TypeName<Int32>() { return std::string(".s32"); }
template<> inline std::string TypeName<Int64>() { return std::string(".s64"); } 

template<> inline std::string TypeName<UInt8>() { return std::string(".u8"); }
template<> inline std::string TypeName<UInt16>() { return std::string(".u16"); }
template<> inline std::string TypeName<UInt32>() { return std::string(".u32"); }
template<> inline std::string TypeName<UInt64>() { return std::string(".u64"); } 

template<> inline std::string TypeName<Float16>() { return std::string(".f16"); }
template<> inline std::string TypeName<Float16x2>() { return std::string(".f16x2"); }
template<> inline std::string TypeName<Float32>() { return std::string(".f32"); }
template<> inline std::string TypeName<Float64>() { return std::string(".f64"); } 

template<> inline std::string TypeName<Bit8>() { return std::string(".b8"); }
template<> inline std::string TypeName<Bit16>() { return std::string(".b16"); }
template<> inline std::string TypeName<Bit32>() { return std::string(".b32"); }
template<> inline std::string TypeName<Bit64>() { return std::string(".b64"); } 

template<> inline std::string TypeName<Predicate>() { return std::string(".pred"); }

}
