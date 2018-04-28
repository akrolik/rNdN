#pragma once

#include <string>

namespace PTX {

enum Bits {
	Bits8  = (1 << 3),
	Bits16 = (1 << 4),
	Bits32 = (1 << 5),
	Bits64 = (1 << 6)
};

enum VectorSize {
	Scalar,
	Vector2,
	Vector4
};

enum VectorElement {
	X,
	Y,
	Z,
	W
};

class Type {}; 

class PredicateType : public Type {};
class VoidType : public Type {};

template<Bits B>
class BitType : public Type {};

template<Bits B>
class IntType : public BitType<B> {};
using Int8Type = IntType<Bits::Bits8>;
using Int16Type = IntType<Bits::Bits16>;
using Int32Type = IntType<Bits::Bits32>;
using Int64Type = IntType<Bits::Bits64>;

template<Bits B>
class UIntType : public BitType<B> {};
using UInt8Type = UIntType<Bits::Bits8>;
using UInt16Type = UIntType<Bits::Bits16>;
using UInt32Type = UIntType<Bits::Bits32>;
using UInt64Type = UIntType<Bits::Bits64>;

template<Bits B>
class FloatType : public BitType<B> {};
using Float16Type = FloatType<Bits::Bits16>;
using Float32Type = FloatType<Bits::Bits32>;
using Float64Type = FloatType<Bits::Bits64>;

template<class T> std::string TypeName()
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
	return ".<unknown>";
}

template<> inline std::string TypeName<IntType<Bits8>>() { return std::string(".s8"); }
template<> inline std::string TypeName<IntType<Bits16>>() { return std::string(".s16"); }
template<> inline std::string TypeName<IntType<Bits32>>() { return std::string(".s32"); }
template<> inline std::string TypeName<IntType<Bits64>>() { return std::string(".s64"); } 

template<> inline std::string TypeName<UIntType<Bits8>>() { return std::string(".u8"); }
template<> inline std::string TypeName<UIntType<Bits16>>() { return std::string(".u16"); }
template<> inline std::string TypeName<UIntType<Bits32>>() { return std::string(".u32"); }
template<> inline std::string TypeName<UIntType<Bits64>>() { return std::string(".u64"); } 

template<> inline std::string TypeName<FloatType<Bits16>>() { return std::string(".f16"); }
// template<> inline std::string TypeName<Float16x2>() { return std::string(".f16x2"); }
template<> inline std::string TypeName<FloatType<Bits32>>() { return std::string(".f32"); }
template<> inline std::string TypeName<FloatType<Bits64>>() { return std::string(".f64"); } 

template<> inline std::string TypeName<BitType<Bits8>>() { return std::string(".b8"); }
template<> inline std::string TypeName<BitType<Bits16>>() { return std::string(".b16"); }
template<> inline std::string TypeName<BitType<Bits32>>() { return std::string(".b32"); }
template<> inline std::string TypeName<BitType<Bits64>>() { return std::string(".b64"); } 

template<> inline std::string TypeName<PredicateType>() { return std::string(".pred"); }

}
