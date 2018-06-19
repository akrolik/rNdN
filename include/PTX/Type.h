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

struct VoidType : Type { static std::string Name() { return ""; } };

struct DataType : Type {};

struct ScalarType : DataType {};

enum class Bits : int {
	Bits1  = (1 << 0),
	Bits8  = (1 << 3),
	Bits16 = (1 << 4),
	Bits32 = (1 << 5),
	Bits64 = (1 << 6)
};

template<Bits B, unsigned int N = 1>
struct BitTypeBase : ScalarType
{
	constexpr static std::underlying_type<Bits>::type BitSize = static_cast<std::underlying_type<Bits>::type>(B);

	static std::string Name() { return ".b" + std::to_string(BitSize); }

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
struct BitTypeBase<Bits::Bits1, 1> : Type
{
	static std::string Name() { return ".pred"; }
};

enum class VectorSize : int;

template<Bits B, unsigned int N = 1> struct BitType : BitTypeBase<B, N> {};
template<> struct BitType<Bits::Bits8, 1> : BitTypeBase<Bits::Bits8>
{
	using SystemType = int8_t;
	constexpr static std::string_view RegisterPrefix = "bc";
};
template<> struct BitType<Bits::Bits16, 1> : BitTypeBase<Bits::Bits16>
{
	using SystemType = int16_t;
	constexpr static std::string_view RegisterPrefix = "bs";
};
template<> struct BitType<Bits::Bits32, 1> : BitTypeBase<Bits::Bits32>
{
	using SystemType = int32_t;
	constexpr static std::string_view RegisterPrefix = "b";
};
template<> struct BitType<Bits::Bits64, 1> : BitTypeBase<Bits::Bits64>
{
	using SystemType = int64_t;
	constexpr static std::string_view RegisterPrefix = "bd";
};

using PredicateType = BitType<Bits::Bits1>;
using Bit8Type = BitType<Bits::Bits8>;
using Bit16Type = BitType<Bits::Bits16>;
using Bit32Type = BitType<Bits::Bits32>;
using Bit64Type = BitType<Bits::Bits64>;

template<Bits B, unsigned int N = 1>
struct IntTypeBase : BitType<B, N>
{
	static_assert(N == 1, "PTX::IntType expects data packing of 1");

	static std::string Name() { return ".s" + std::to_string(BitType<B, N>::BitSize); }

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

template<Bits B, unsigned int N = 1> struct IntType : IntTypeBase<B, N> {};
template<> struct IntType<Bits::Bits8, 1> : IntTypeBase<Bits::Bits8>
{
	using SystemType = int8_t;
	constexpr static std::string_view RegisterPrefix = "rc";
};
template<> struct IntType<Bits::Bits16, 1> : IntTypeBase<Bits::Bits16>
{
	using SystemType = int16_t;
	constexpr static std::string_view RegisterPrefix = "rs";
};
template<> struct IntType<Bits::Bits32, 1> : IntTypeBase<Bits::Bits32>
{
	using SystemType = int32_t;
	constexpr static std::string_view RegisterPrefix = "r";
};
template<> struct IntType<Bits::Bits64, 1> : IntTypeBase<Bits::Bits64>
{
	using SystemType = int64_t;
	constexpr static std::string_view RegisterPrefix = "rd";
};

using Int8Type = IntType<Bits::Bits8>;
using Int16Type = IntType<Bits::Bits16>;
using Int32Type = IntType<Bits::Bits32>;
using Int64Type = IntType<Bits::Bits64>;

template<Bits B, unsigned int N = 1>
struct UIntTypeBase : BitType<B, N>
{
	static_assert(N == 1, "PTX::UIntType expects data packing of 1");

	static std::string Name() { return ".u" + std::to_string(BitType<B, N>::BitSize); }

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

template<Bits B, unsigned int N = 1> struct UIntType : UIntTypeBase<B, N> {};
template<> struct UIntType<Bits::Bits8, 1> : UIntTypeBase<Bits::Bits8>
{
	using SystemType = uint8_t;
	constexpr static std::string_view RegisterPrefix = "uc";
};
template<> struct UIntType<Bits::Bits16, 1> : UIntTypeBase<Bits::Bits16>
{
	using SystemType = uint16_t;
	constexpr static std::string_view RegisterPrefix = "us";
};
template<> struct UIntType<Bits::Bits32, 1> : UIntTypeBase<Bits::Bits32>
{
	using SystemType = uint32_t;
	constexpr static std::string_view RegisterPrefix = "u";
};
template<> struct UIntType<Bits::Bits64, 1> : UIntTypeBase<Bits::Bits64>
{
	using SystemType = uint64_t;
	constexpr static std::string_view RegisterPrefix = "ud";
};

using UInt8Type = UIntType<Bits::Bits8>;
using UInt16Type = UIntType<Bits::Bits16>;
using UInt32Type = UIntType<Bits::Bits32>;
using UInt64Type = UIntType<Bits::Bits64>;

template<Bits B, unsigned int N = 1>
struct FloatTypeBase : BitType<B, N>
{
	static_assert(N == 1, "PTX::FloatType expects data packing of 1");

	static std::string Name() { return ".f" + std::to_string(BitType<B, N>::BitSize); }

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
struct FloatTypeBase<Bits::Bits16, N> : BitType<Bits::Bits16, N>
{
	static std::string Name()
	{
		if constexpr(N == 1)
			return ".f16";
		else
			return ".f16x" + std::to_string(N);
	}

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

template<Bits B, unsigned int N = 1> struct FloatType : FloatTypeBase<B, N> {};
template<> struct FloatType<Bits::Bits16, 1> : FloatTypeBase<Bits::Bits16>
{
	using SystemType = float;
	constexpr static std::string_view RegisterPrefix = "h";
};
template<> struct FloatType<Bits::Bits16, 2> : FloatTypeBase<Bits::Bits16>
{
	constexpr static std::string_view RegisterPrefix = "hh";
};
template<> struct FloatType<Bits::Bits32, 1> : FloatTypeBase<Bits::Bits32>
{
	using SystemType = float;
	constexpr static std::string_view RegisterPrefix = "f";
};
template<> struct FloatType<Bits::Bits64, 1> : FloatTypeBase<Bits::Bits64>
{
	using SystemType = double;
	constexpr static std::string_view RegisterPrefix = "fd";
};

using Float16Type = FloatType<Bits::Bits16>;
using Float16x2Type = FloatType<Bits::Bits16, 2>;
using Float32Type = FloatType<Bits::Bits32>;
using Float64Type = FloatType<Bits::Bits64>;

enum class VectorSize : int {
	Vector2 = 2,
	Vector4 = 4
};

template<VectorSize V> std::string VectorName() { return std::string(".<unknown>"); }
template<> inline std::string VectorName<VectorSize::Vector2>() { return std::string(".v2"); }
template<> inline std::string VectorName<VectorSize::Vector4>() { return std::string(".v4"); }

template<class T, VectorSize V>
struct VectorType : DataType
{
	REQUIRE_BASE_TYPE(VectorType, ScalarType);

	using ElementType = T;
	constexpr static std::underlying_type<VectorSize>::type ElementCount = static_cast<std::underlying_type<VectorSize>::type>(V);

	static std::string Name() { return ".v" + std::to_string(ElementCount) + " " + T::Name(); }
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
struct PointerType : UIntType<B>
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
