#pragma once

#include <string>

#include "PTX/StateSpace.h"
#include "PTX/Utils.h"

#define __TO_STRING(S) #S
#define TO_STRING(S) __TO_STRING(S)

namespace PTX {

// @struct is_rounding_type
//
// Type trait for determining if a PTX::Type has a rounding mode

template <class T, typename E = void>
struct is_rounding_type : std::false_type {};

template <class T>
struct is_rounding_type<T, std::enable_if_t<std::is_enum<typename T::RoundingMode>::value>> : std::true_type {};

// @struct is_type_specialization
//
// Type trait for determining if a type is a specialization of some template

enum class Bits : int;

template <class T, template <Bits, unsigned int> class Template>
struct is_type_specialization : std::false_type {};

template <template <Bits, unsigned int> class Template, Bits B, unsigned int N>
struct is_type_specialization<Template<B, N>, Template> : std::true_type {};

template<class T, class... R>
struct TypeEnforcer
{
	constexpr static bool value = is_one<std::is_same<R, T>::value...>::value;
};

#define REQUIRE_TYPE_PARAMS(context, D_ENABLED, T_ENABLED) \
	constexpr static bool Enabled = D_ENABLED && T_ENABLED; \
	static_assert(Typecheck == false || Enabled == true, "PTX::" TO_STRING(context) " does not support types PTX::" TO_STRING(D) " and PTX::" TO_STRING(T));

#define REQUIRE_TYPE_PARAM(P, ...) \
	TypeEnforcer<P, __VA_ARGS__>::value

#define REQUIRE_TYPE(context, ...) \
	constexpr static bool Enabled = REQUIRE_TYPE_PARAM(T, __VA_ARGS__); \
	static_assert(Typecheck == false || Enabled == true, "PTX::" TO_STRING(context) " does not support type PTX::" TO_STRING(T));

#define REQUIRE_BASE_TYPE(context, type) static_assert(std::is_base_of<type, T>::value, "PTX::" TO_STRING(context) " requires base type PTX::" TO_STRING(type))
#define REQUIRE_EXACT_TYPE(context, type) static_assert(std::is_same<type, T>::value, "PTX::" TO_STRING(context) " requires exact type PTX::" TO_STRING(type))
#define REQUIRE_EXACT_TYPE_TEMPLATE(context, type) static_assert(is_type_specialization<T, type>::value, "PTX::" TO_STRING(context) " requires exact type template PTX::" TO_STRING(type))

// @struct Type
//
// Type
//   VoidType
//   PredicateType
//   DataType
//     ScalarType
//       BitType<Bits>
//         IntType
//         UIntType
//           PointerType<AddressSpace>
//         FloatType
//         PackedType<T, N>
//     VectorType<ScalarType, VectorSize>

struct Type { static std::string Name() { return ".<unknown>"; } }; 

struct VoidType : private Type { static std::string Name() { return ""; } };

struct DataType : private Type {};

struct ScalarType : private DataType {};

enum class Bits : int {
	Bits1  = (1 << 0),
	Bits8  = (1 << 3),
	Bits16 = (1 << 4),
	Bits32 = (1 << 5),
	Bits64 = (1 << 6)
};

template<Bits B, unsigned int N = 1>
struct BitType : private ScalarType
{
	static std::string Name() { return ".b" + std::to_string(static_cast<std::underlying_type<Bits>::type>(B)); }

	enum class ComparisonOperator {
		Equal,
		NotEqual
	};

	static std::string ComparisonOperatorString(ComparisonOperator comparisonOperator)
	{
		switch (comparisonOperator)
		{
			case ComparisonOperator::Equal:
				return ".eq";
			case ComparisonOperator::NotEqual:
				return ".ne";
		}
		return "";
	}
};

template<>
struct BitType<Bits::Bits1, 1> : private Type
{
	static std::string Name() { return ".pred"; }
};

using PredicateType = BitType<Bits::Bits1>;
using Bit8Type = BitType<Bits::Bits8>;
using Bit16Type = BitType<Bits::Bits16>;
using Bit32Type = BitType<Bits::Bits32>;
using Bit64Type = BitType<Bits::Bits64>;

template<Bits B, unsigned int N = 1>
struct IntTypeBase : private BitType<B, N>
{
	static_assert(N == 1, "PTX::IntType expects data packing of 1");

	static const bool CarryModifier = (B == Bits::Bits32 || B == Bits::Bits64);
	static const bool HalfModifier = true;
	static const bool FlushModifier = false;
	static const bool SaturateModifier = (B == Bits::Bits32);

	static std::string Name() { return ".s" + std::to_string(static_cast<std::underlying_type<Bits>::type>(B)); }

	enum class ComparisonOperator {
		Equal,
		NotEqual,
		Less,
		LessEqual,
		Greater,
		GreaterEqual
	};

	static std::string ComparisonOperatorString(ComparisonOperator comparisonOperator)
	{
		switch (comparisonOperator)
		{
			case ComparisonOperator::Equal:
				return ".eq";
			case ComparisonOperator::NotEqual:
				return ".ne";
			case ComparisonOperator::Less:
				return ".lt";
			case ComparisonOperator::LessEqual:
				return ".le";
			case ComparisonOperator::Greater:
				return ".gt";
			case ComparisonOperator::GreaterEqual:
				return ".ge";
		}
		return "";
	}
};

template<Bits B, unsigned int N = 1> struct IntType : public IntTypeBase<B, N> {};
template<> struct IntType<Bits::Bits8, 1> : public IntTypeBase<Bits::Bits8> { using SystemType = int8_t; };
template<> struct IntType<Bits::Bits16, 1> : public IntTypeBase<Bits::Bits16> { using SystemType = int16_t; };
template<> struct IntType<Bits::Bits32, 1> : public IntTypeBase<Bits::Bits32> { using SystemType = int32_t; };
template<> struct IntType<Bits::Bits64, 1> : public IntTypeBase<Bits::Bits64> { using SystemType = int64_t; };

using Int8Type = IntType<Bits::Bits8>;
using Int16Type = IntType<Bits::Bits16>;
using Int32Type = IntType<Bits::Bits32>;
using Int64Type = IntType<Bits::Bits64>;

template<Bits B, unsigned int N = 1>
struct UIntTypeBase : private BitType<B, N>
{
	static_assert(N == 1, "PTX::UIntType expects data packing of 1");

	static std::string Name() { return ".u" + std::to_string(static_cast<std::underlying_type<Bits>::type>(B)); }

	static const bool CarryModifier = (B == Bits::Bits32 || B == Bits::Bits64);
	static const bool HalfModifier = true;
	static const bool FlushModifier = false;
	static const bool SaturateModifier = false;

	enum class ComparisonOperator {
		Equal,
		NotEqual,
		Lower,
		LowerSame,
		Higher,
		HigherSame
	};

	static std::string ComparisonOperatorString(ComparisonOperator comparisonOperator)
	{
		switch (comparisonOperator)
		{
			case ComparisonOperator::Equal:
				return ".eq";
			case ComparisonOperator::NotEqual:
				return ".ne";
			case ComparisonOperator::Lower:
				return ".lo";
			case ComparisonOperator::LowerSame:
				return ".ls";
			case ComparisonOperator::Higher:
				return ".hi";
			case ComparisonOperator::HigherSame:
				return ".hs";
		}
		return "";
	}
};

template<Bits B, unsigned int N = 1> struct UIntType : public UIntTypeBase<B, N> {};
template<> struct UIntType<Bits::Bits8, 1> : public UIntTypeBase<Bits::Bits8> { using SystemType = uint8_t; };
template<> struct UIntType<Bits::Bits16, 1> : public UIntTypeBase<Bits::Bits16> { using SystemType = uint16_t; };
template<> struct UIntType<Bits::Bits32, 1> : public UIntTypeBase<Bits::Bits32> { using SystemType = uint32_t; };
template<> struct UIntType<Bits::Bits64, 1> : public UIntTypeBase<Bits::Bits64> { using SystemType = uint64_t; };

using UInt8Type = UIntType<Bits::Bits8>;
using UInt16Type = UIntType<Bits::Bits16>;
using UInt32Type = UIntType<Bits::Bits32>;
using UInt64Type = UIntType<Bits::Bits64>;

template<Bits B, unsigned int N = 1>
struct FloatTypeBase : private BitType<B, N>
{
	static_assert(N == 1, "PTX::FloatType expects data packing of 1");

	static std::string Name() { return ".f" + std::to_string(static_cast<std::underlying_type<Bits>::type>(B)); }

	static const bool CarryModifier = false;
	static const bool HalfModifier = false;
	static const bool FlushModifier = (B == Bits::Bits32);
	static const bool SaturateModifier = (B == Bits::Bits32);

	enum class RoundingMode {
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
			case RoundingMode::Nearest:
				return ".rn";
			case RoundingMode::Zero:
				return ".rz";
			case RoundingMode::NegativeInfinity:
				return ".rm";
			case RoundingMode::PositiveInfinity:
				return ".rp";
		}
		return "";
	}

	enum class ComparisonOperator {
		Equal,
		NotEqual,
		Less,
		LessEqual,
		Greater,
		GreaterEqual,
		
		EqualUnordered,
		NotEqualUnordered,
		LessUnordered,
		LessEqualUnordered,
		GreaterUnordered,
		GreaterEqualUnordered,

		Number,
		NaN
	};

	static std::string ComparisonOperatorString(ComparisonOperator comparisonOperator)
	{
		switch (comparisonOperator)
		{
			case ComparisonOperator::Equal:
				return ".eq";
			case ComparisonOperator::NotEqual:
				return ".ne";
			case ComparisonOperator::Less:
				return ".lt";
			case ComparisonOperator::LessEqual:
				return ".le";
			case ComparisonOperator::Greater:
				return ".gt";
			case ComparisonOperator::GreaterEqual:
				return ".ge";
			case ComparisonOperator::EqualUnordered:
				return ".equ";
			case ComparisonOperator::NotEqualUnordered:
				return ".neu";
			case ComparisonOperator::LessUnordered:
				return ".ltu";
			case ComparisonOperator::LessEqualUnordered:
				return ".leu";
			case ComparisonOperator::GreaterUnordered:
				return ".gtu";
			case ComparisonOperator::GreaterEqualUnordered:
				return ".geu";
			case ComparisonOperator::Number:
				return ".num";
			case ComparisonOperator::NaN:
				return ".nan";
		}
		return "";
	}
};

template<unsigned int N>
struct FloatTypeBase<Bits::Bits16, N> : private BitType<Bits::Bits16, N>
{
	static std::string Name()
	{
		if constexpr(N == 1)
			return ".f16";
		else
			return ".f16x" + std::to_string(N);
	}

	static const bool CarryModifier = false;
	static const bool HalfModifier = false;
	static const bool FlushModifier = true;
	static const bool SaturateModifier = true;

	enum class RoundingMode {
		None,
		Nearest,
	};

	static std::string RoundingModeString(RoundingMode roundingMode)
	{
		if (roundingMode == RoundingMode::None)
		{
			return "";
		}
		return ".rn";
	}

	enum class ComparisonOperator {
		Equal,
		NotEqual,
		Less,
		LessEqual,
		Greater,
		GreaterEqual,
		
		EqualUnordered,
		NotEqualUnordered,
		LessUnordered,
		LessEqualUnordered,
		GreaterUnordered,
		GreaterEqualUnordered,

		Number,
		NaN
	};

	static std::string ComparisonOperatorString(ComparisonOperator comparisonOperator)
	{
		switch (comparisonOperator)
		{
			case ComparisonOperator::Equal:
				return ".eq";
			case ComparisonOperator::NotEqual:
				return ".ne";
			case ComparisonOperator::Less:
				return ".lt";
			case ComparisonOperator::LessEqual:
				return ".le";
			case ComparisonOperator::Greater:
				return ".gt";
			case ComparisonOperator::GreaterEqual:
				return ".ge";
			case ComparisonOperator::EqualUnordered:
				return ".equ";
			case ComparisonOperator::NotEqualUnordered:
				return ".neu";
			case ComparisonOperator::LessUnordered:
				return ".ltu";
			case ComparisonOperator::LessEqualUnordered:
				return ".leu";
			case ComparisonOperator::GreaterUnordered:
				return ".gtu";
			case ComparisonOperator::GreaterEqualUnordered:
				return ".geu";
			case ComparisonOperator::Number:
				return ".num";
			case ComparisonOperator::NaN:
				return ".nan";
		}
		return "";
	}
};

template<Bits B, unsigned int N = 1> struct FloatType : public FloatTypeBase<B, N> {};
template<> struct FloatType<Bits::Bits16, 1> : public FloatTypeBase<Bits::Bits16> { using SystemType = float; };
template<> struct FloatType<Bits::Bits32, 1> : public FloatTypeBase<Bits::Bits32> { using SystemType = float; };
template<> struct FloatType<Bits::Bits64, 1> : public FloatTypeBase<Bits::Bits64> { using SystemType = double; };

using Float16Type = FloatType<Bits::Bits16>;
using Float32Type = FloatType<Bits::Bits32>;
using Float64Type = FloatType<Bits::Bits64>;
using Float16x2Type = FloatType<Bits::Bits16, 2>;

enum class VectorSize : int {
	Vector2 = 2,
	Vector4 = 4
};

template<VectorSize V> std::string VectorName() { return std::string(".<unknown>"); }
template<> inline std::string VectorName<VectorSize::Vector2>() { return std::string(".v2"); }
template<> inline std::string VectorName<VectorSize::Vector4>() { return std::string(".v4"); }

template<class T, VectorSize V>
struct VectorType : private DataType
{
	REQUIRE_BASE_TYPE(VectorType, ScalarType);

	static std::string Name() { return ".v" + std::to_string(static_cast<std::underlying_type<VectorSize>::type>(V)) + " " + T::Name(); }
};

template<class T>
using Vector2Type = VectorType<T, VectorSize::Vector2>;
template<class T>
using Vector4Type = VectorType<T, VectorSize::Vector4>;

enum class VectorElement {
	X,
	Y,
	Z,
	W
};

static std::string GetVectorElementName(VectorElement vectorElement)
{
	switch (vectorElement)
	{
		case VectorElement::X:
			return ".x";
		case VectorElement::Y:
			return ".y";
		case VectorElement::Z:
			return ".z";
		case VectorElement::W:
			return ".w";
	}
	return ".<unknown>";
}

template<class T, Bits B, class S = AddressableSpace>
struct PointerType : private UIntType<B>
{
	REQUIRE_BASE_TYPE(PointerType, DataType);
	REQUIRE_BASE_SPACE(PointerType, AddressableSpace);

	static std::string Name() { return UIntType<B>::Name(); }
};

template<class T, class S = AddressableSpace>
using Pointer32Type = PointerType<T, Bits::Bits32, S>;
template<class T, class S = AddressableSpace>
using Pointer64Type = PointerType<T, Bits::Bits64, S>;

}
