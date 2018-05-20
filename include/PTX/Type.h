#pragma once

#include <string>

#define __TO_STRING(S) #S
#define TO_STRING(S) __TO_STRING(S)

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

struct VoidType : public Type {};

struct ValueType : public Type {};

struct PredicateType : public ValueType { static std::string Name() { return ".pred"; } };

struct DataType : public ValueType {};

struct ScalarType : public DataType {};

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
struct FloatType : public BitType<B>
{
	static std::string Name() { return ".f" + std::to_string(B); }

	enum RoundingMode {
		None,
		Nearest,
		Zero,
		NegativeInfinity,
		PositiveInfinity
	};

	static std::string RoundingModeString(RoundingMode roundingMode)
	{
		switch (roundingMode)
		{
			case Nearest:
				return ".rn";
			case Zero:
				return ".rz";
			case NegativeInfinity:
				return ".rm";
			case PositiveInfinity:
				return ".rp";
		}
		return "";
	}
};

template<>
struct FloatType<Bits::Bits16> : public BitType<Bits::Bits16>
{
	static std::string Name() { return ".f16"; }

	enum RoundingMode {
		None,
		Nearest,
	};

	static std::string RoundingModeString(RoundingMode roundingMode)
	{
		if (roundingMode == None)
		{
			return "";
		}
		return ".rn";
	}
};

using Float16Type = FloatType<Bits::Bits16>;
using Float32Type = FloatType<Bits::Bits32>;
using Float64Type = FloatType<Bits::Bits64>;

template<template<Bits B> class T, Bits B, unsigned int N>
struct PackedType : public BitType<Bits(B * N)>
{
	static std::string Name() { return T<B>::Name() + "x" + std::to_string(N); }
};

template<unsigned int N>
struct PackedType<FloatType, Bits::Bits16, N>
{
	static std::string Name() { return Float16Type::Name() + "x" + std::to_string(N); }

	enum RoundingMode {
		None,
		Nearest
	};

	static std::string RoundingModeString(RoundingMode roundingMode)
	{
		if (roundingMode == None)
		{
			return "";
		}
		return ".rn";
	}
};

using Float16x2 = PackedType<FloatType, Bits::Bits16, 2>;

template <class T, template <Bits> class Template>
struct is_type_specialization : std::false_type {};

template <template <Bits> class Template, Bits Args>
struct is_type_specialization<Template<Args>, Template> : std::true_type {};
 
#define DISABLE_ALL(inst, type) static_assert(std::is_same<type, T>::value && !std::is_same<type, T>::value, "PTX::" TO_STRING(inst) " does not support PTX::" TO_STRING(type))

#define DISABLE_TYPE(inst, type) static_assert(!std::is_same<type, T>::value, "PTX::" TO_STRING(inst) " does not support PTX::" TO_STRING(type))
#define DISABLE_TYPES(inst, type) static_assert(!is_type_specialization<T, type>::value, "PTX::" TO_STRING(inst) " does not support PTX::" TO_STRING(type))
#define DISABLE_TYPE_BITS(inst, type, bits) static_assert(B != bits, "PTX::" TO_STRING(inst) " does not support PTX::" TO_STRING(type) " with PTX::Bits::" TO_STRING(bits))
#define DISABLE_BITS(inst, bits) static_assert(B != bits, "PTX::" TO_STRING(inst) " does not support PTX::Bits::" TO_STRING(bits))

#define REQUIRE_TYPE(inst, type) static_assert(std::is_base_of<type, T>::value, "PTX::" TO_STRING(inst) " must be a PTX::" TO_STRING(type))
#define REQUIRE_TYPES(inst, type) static_assert(is_type_specialization<T, type>::value, "PTX::" TO_STRING(inst) " must be a PTX::" TO_STRING(type))

enum VectorSize {
	Vector2 = 2,
	Vector4 = 4
};

template<VectorSize V> std::string VectorName() { return std::string(".<unknown>"); }
template<> inline std::string VectorName<Vector2>() { return std::string(".v2"); }
template<> inline std::string VectorName<Vector4>() { return std::string(".v4"); }

template<class T, VectorSize V>
struct VectorType : public DataType
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

}
