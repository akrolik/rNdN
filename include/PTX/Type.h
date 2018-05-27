#pragma once

#include <string>

#define __TO_STRING(S) #S
#define TO_STRING(S) __TO_STRING(S)

namespace PTX {

// @enum AddressSpace
//
// Storage space used for addressable state spaces

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

enum Bits : int;

template <class T, template <Bits, unsigned int> class Template>
struct is_type_specialization : std::false_type {};

template <template <Bits, unsigned int> class Template, Bits B, unsigned int N>
struct is_type_specialization<Template<B, N>, Template> : std::true_type {};
 
#define DISABLE_EXACT_TYPE(context, type) static_assert(!std::is_same<type, T>::value, "PTX::" TO_STRING(context) " does not support type PTX::" TO_STRING(type))
#define DISABLE_EXACT_TYPE_TEMPLATE(context, template) static_assert(!is_type_specialization<T, template>::value, "PTX::" TO_STRING(context) " does not support type template PTX::" TO_STRING(template)) 

#define REQUIRE_BASE_TYPE(context, type) static_assert(std::is_base_of<type, T>::value, "PTX::" TO_STRING(context) " requires base type PTX::" TO_STRING(type))
#define REQUIRE_EXACT_TYPE(context, type) static_assert(std::is_same<type, T>::value, "PTX::" TO_STRING(context) " requires exact type PTX::" TO_STRING(type))
#define REQUIRE_EXACT_TYPE_TEMPLATE(context, type) static_assert(is_type_specialization<T, type>::value, "PTX::" TO_STRING(context) " requires exact type template PTX::" TO_STRING(type))

// @struct Type
//
// Type
//   VoidType
//   DataType
//     ScalarType
//       BitType<Bits>
//         PredicateType
//         IntType
//         UIntType
//           PointerType<AddressSpace>
//         FloatType
//         PackedType<T, N>
//     VectorType<ScalarType, VectorSize>

struct Type { static std::string Name() { return ".<unknown>"; } }; 

//TODO: temporary ToString for function printing
struct VoidType : private Type { static std::string Name() { return ""; }  std::string ToString() { return ""; } };

struct DataType : private Type {};

struct ScalarType : private DataType {};

enum Bits : int {
	Bits1  = (1 << 0),
	Bits8  = (1 << 3),
	Bits16 = (1 << 4),
	Bits32 = (1 << 5),
	Bits64 = (1 << 6)
};

template<Bits B, unsigned int N = 1>
struct BitType : private ScalarType
{
	static std::string Name() { return ".b" + std::to_string(B); }

	enum ComparisonOperator {
		Equal,
		NotEqual
	};

	static std::string ComparisonOperatorString(ComparisonOperator comparisonOperator)
	{
		switch (comparisonOperator)
		{
			case Equal:
				return ".eq";
			case NotEqual:
				return ".ne";
		}
		return "";
	}
};

template<>
struct BitType<Bits::Bits1, 1> : private ScalarType
{
	static std::string Name() { return ".pred"; }
};

using PredicateType = BitType<Bits::Bits1>;
using Bit8Type = BitType<Bits::Bits8>;
using Bit16Type = BitType<Bits::Bits16>;
using Bit32Type = BitType<Bits::Bits32>;
using Bit64Type = BitType<Bits::Bits64>;

template<Bits B, unsigned int N = 1>
struct IntType : private BitType<B, N>
{
	static_assert(N == 1, "PTX::IntType expects data packing of 1");

	static const bool CarryModifier = (B == Bits::Bits32 || B == Bits::Bits64);
	static const bool HalfModifier = true;
	static const bool FlushModifier = false;
	static const bool SaturateModifier = (B == Bits::Bits32);

	static std::string Name() { return ".s" + std::to_string(B); }

	enum ComparisonOperator {
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
			case Equal:
				return ".eq";
			case NotEqual:
				return ".ne";
			case Less:
				return ".lt";
			case LessEqual:
				return ".le";
			case Greater:
				return ".gt";
			case GreaterEqual:
				return ".ge";
		}
		return "";
	}
};

using Int8Type = IntType<Bits::Bits8>;
using Int16Type = IntType<Bits::Bits16>;
using Int32Type = IntType<Bits::Bits32>;
using Int64Type = IntType<Bits::Bits64>;

template<Bits B, unsigned int N = 1>
struct UIntType : private BitType<B, N>
{
	static_assert(N == 1, "PTX::UIntType expects data packing of 1");

	static std::string Name() { return ".u" + std::to_string(B); }

	static const bool CarryModifier = (B == Bits::Bits32 || B == Bits::Bits64);
	static const bool HalfModifier = true;
	static const bool FlushModifier = false;
	static const bool SaturateModifier = false;

	enum ComparisonOperator {
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
			case Equal:
				return ".eq";
			case NotEqual:
				return ".ne";
			case Lower:
				return ".lo";
			case LowerSame:
				return ".ls";
			case Higher:
				return ".hi";
			case HigherSame:
				return ".hs";
		}
		return "";
	}
};

using UInt8Type = UIntType<Bits::Bits8>;
using UInt16Type = UIntType<Bits::Bits16>;
using UInt32Type = UIntType<Bits::Bits32>;
using UInt64Type = UIntType<Bits::Bits64>;

template<Bits B, unsigned int N = 1>
struct FloatType : private BitType<B, N>
{
	static_assert(N == 1, "PTX::FloatType expects data packing of 1");

	static std::string Name() { return ".f" + std::to_string(B); }

	static const bool CarryModifier = false;
	static const bool HalfModifier = false;
	static const bool FlushModifier = (B == Bits::Bits32);
	static const bool SaturateModifier = (B == Bits::Bits32);

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

	enum ComparisonOperator {
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
			case Equal:
				return ".eq";
			case NotEqual:
				return ".ne";
			case Less:
				return ".lt";
			case LessEqual:
				return ".le";
			case Greater:
				return ".gt";
			case GreaterEqual:
				return ".ge";
			case EqualUnordered:
				return ".equ";
			case NotEqualUnordered:
				return ".neu";
			case LessUnordered:
				return ".ltu";
			case LessEqualUnordered:
				return ".leu";
			case GreaterUnordered:
				return ".gtu";
			case GreaterEqualUnordered:
				return ".geu";
			case Number:
				return ".num";
			case NaN:
				return ".nan";
		}
		return "";
	}
};

template<unsigned int N>
struct FloatType<Bits::Bits16, N> : private BitType<Bits::Bits16, N>
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

	enum ComparisonOperator {
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
			case Equal:
				return ".eq";
			case NotEqual:
				return ".ne";
			case Less:
				return ".lt";
			case LessEqual:
				return ".le";
			case Greater:
				return ".gt";
			case GreaterEqual:
				return ".ge";
			case EqualUnordered:
				return ".equ";
			case NotEqualUnordered:
				return ".neu";
			case LessUnordered:
				return ".ltu";
			case LessEqualUnordered:
				return ".leu";
			case GreaterUnordered:
				return ".gtu";
			case GreaterEqualUnordered:
				return ".geu";
			case Number:
				return ".num";
			case NaN:
				return ".nan";
		}
		return "";
	}
};

using Float16Type = FloatType<Bits::Bits16>;
using Float32Type = FloatType<Bits::Bits32>;
using Float64Type = FloatType<Bits::Bits64>;
using Float16x2Type = FloatType<Bits::Bits16, 2>;

enum VectorSize {
	Vector2 = 2,
	Vector4 = 4
};

template<VectorSize V> std::string VectorName() { return std::string(".<unknown>"); }
template<> inline std::string VectorName<Vector2>() { return std::string(".v2"); }
template<> inline std::string VectorName<Vector4>() { return std::string(".v4"); }

template<class T, VectorSize V>
struct VectorType : private DataType
{
	REQUIRE_BASE_TYPE(VectorType, ScalarType);
	DISABLE_EXACT_TYPE(VectorType, PredicateType);

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

template<class T, Bits B, AddressSpace A = AddressSpace::Generic>
struct PointerType : private UIntType<B>
{
	REQUIRE_BASE_TYPE(PointerType, Type);
	DISABLE_EXACT_TYPE(PointerType, PredicateType);

	static std::string Name() { return UIntType<B>::Name(); }
};

template<class T, AddressSpace A = AddressSpace::Generic>
using Pointer32Type = PointerType<T, Bits::Bits32, A>;
template<class T, AddressSpace A = AddressSpace::Generic>
using Pointer64Type = PointerType<T, Bits::Bits64, A>;

}
