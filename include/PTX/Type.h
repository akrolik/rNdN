#pragma once

#include <string>

namespace PTX {

enum Bits {
	Bits8  = (1 << 3),
	Bits16 = (1 << 4),
	Bits32 = (1 << 5),
	Bits64 = (1 << 6)
};

enum AddressSpace {
	Generic,
	Const,
	Global,
	Local,
	Param,
	Shared
};

template<AddressSpace S> std::string AddressSpaceName() { return ".<unknown>"; }

template<> inline std::string AddressSpaceName<Generic>() { return std::string(""); }
template<> inline std::string AddressSpaceName<Const>() { return std::string(".const"); }
template<> inline std::string AddressSpaceName<Global>() { return std::string(".global"); }
template<> inline std::string AddressSpaceName<Local>() { return std::string(".local"); }
template<> inline std::string AddressSpaceName<Param>() { return std::string(".param"); }
template<> inline std::string AddressSpaceName<Shared>() { return std::string(".shared"); }

static std::string GetAddressSpaceName(AddressSpace addressSpace)
{
	switch (addressSpace)
	{
		case Generic:
			return "";
		case Const:
			return ".const";
		case Global:
			return ".global";
		case Local:
			return ".local";
		case Param:
			return ".param";
	}
	return ".<unknown>";
}

struct Type { static std::string Name() { return ".<unknown>"; } }; 

struct PredicateType : public Type { static std::string Name() { return ".pred"; } };
struct VoidType : public Type {};

struct ScalarType : public Type {};

template<Bits B>
struct BitType : public ScalarType {};

template<Bits B>
struct IntType : public BitType<B> { static std::string Name() { return ".s" + std::to_string(B); } };
using Int8Type = IntType<Bits::Bits8>;
using Int16Type = IntType<Bits::Bits16>;
using Int32Type = IntType<Bits::Bits32>;
using Int64Type = IntType<Bits::Bits64>;

template<Bits B>
struct UIntType : public BitType<B> { static std::string Name() { return ".u" + std::to_string(B); } };
using UInt8Type = UIntType<Bits::Bits8>;
using UInt16Type = UIntType<Bits::Bits16>;
using UInt32Type = UIntType<Bits::Bits32>;
using UInt64Type = UIntType<Bits::Bits64>;

template<Bits B>
struct FloatType : public BitType<B> { static std::string Name() { return ".f" + std::to_string(B); } };
using Float16Type = FloatType<Bits::Bits16>;
using Float32Type = FloatType<Bits::Bits32>;
using Float64Type = FloatType<Bits::Bits64>;

enum VectorSize {
	Vector2 = 2,
	Vector4 = 4
};

template<VectorSize V> std::string VectorName() { return std::string(".<unknown>"); }
template<> inline std::string VectorName<Vector2>() { return std::string(".v2"); }
template<> inline std::string VectorName<Vector4>() { return std::string(".v4"); }

template<class T, VectorSize V>
struct VectorType : public Type
{
	static_assert(std::is_base_of<ScalarType, T>::value, "T must be a PTX::ScalarType");

	static std::string Name() { return ".v" + std::to_string(V) + " " + T::Name(); }
};
template<class T>
using Vector2Type = VectorType<T, Vector2>;
template<class T>
using Vector4Type = VectorType<T, Vector4>;

enum VectorElement {
	X,
	Y,
	Z,
	W
};

static std::string GetVectorElementName(VectorElement vectorElement)
{
	switch (vectorElement)
	{
		case X:
			return ".x";
		case Y:
			return ".y";
		case Z:
			return ".z";
		case W:
			return ".w";
	}
	return ".<unknown>";
}
 
// template<class T> std::string TypeName()
// {
// 	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
// 	return ".<unknown>";
// }

// template<> inline std::string TypeName<PredicateType>() { return std::string(".pred"); }

// template<> inline std::string TypeName<IntType<Bits8>>() { return std::string(".s8"); }
// template<> inline std::string TypeName<IntType<Bits16>>() { return std::string(".s16"); }
// template<> inline std::string TypeName<IntType<Bits32>>() { return std::string(".s32"); }
// template<> inline std::string TypeName<IntType<Bits64>>() { return std::string(".s64"); } 

// template<> inline std::string TypeName<UIntType<Bits8>>() { return std::string(".u8"); }
// template<> inline std::string TypeName<UIntType<Bits16>>() { return std::string(".u16"); }
// template<> inline std::string TypeName<UIntType<Bits32>>() { return std::string(".u32"); }
// template<> inline std::string TypeName<UIntType<Bits64>>() { return std::string(".u64"); } 

// template<> inline std::string TypeName<FloatType<Bits16>>() { return std::string(".f16"); }
// // template<> inline std::string TypeName<Float16x2>() { return std::string(".f16x2"); }
// template<> inline std::string TypeName<FloatType<Bits32>>() { return std::string(".f32"); }
// template<> inline std::string TypeName<FloatType<Bits64>>() { return std::string(".f64"); } 

// template<> inline std::string TypeName<BitType<Bits8>>() { return std::string(".b8"); }
// template<> inline std::string TypeName<BitType<Bits16>>() { return std::string(".b16"); }
// template<> inline std::string TypeName<BitType<Bits32>>() { return std::string(".b32"); }
// template<> inline std::string TypeName<BitType<Bits64>>() { return std::string(".b64"); } 

// template<class T, VectorSize V>
// template<> inline std::string TypeName<VectorType<T, V>>() { return VectorName<V>() + " " + TypeName<T>(); }

}
