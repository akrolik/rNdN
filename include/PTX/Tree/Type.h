#pragma once

#include <string>

#include "PTX/Tree/Concepts.h"
#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Utils.h"

#define __TO_STRING(S) #S
#define TO_STRING(S) __TO_STRING(S)

namespace PTX {

#define REQUIRE_TYPE_PARAM(CONTEXT, ENABLED) \
	constexpr static bool TypeSupported = ENABLED; \
	static_assert(Assert == false || TypeSupported == true, "PTX::" TO_STRING(CONTEXT) " does not support PTX type");

#define REQUIRE_TYPE_PARAMS(context, D_ENABLED, T_ENABLED) \
	constexpr static bool TypeSupported = D_ENABLED && T_ENABLED; \
	static_assert(Assert == false || TypeSupported == true, "PTX::" TO_STRING(context) " does not support PTX types");

// @enum Bits
//
// Enumeration of the number of bits in a single data element (does not include
// packing or vector information)

enum class Bits : int {
	Bits1  = (1 << 0),
	Bits8  = (1 << 3),
	Bits16 = (1 << 4),
	Bits32 = (1 << 5),
	Bits64 = (1 << 6)
};

// @struct Type
//
// Type
//   VoidType
//   DataType
//     ArrayType<DataType, N>
//     ValueType
//       ScalarType
//         BitType<Bits, N>
//           IntType
//           UIntType
//             PointerType<AddressSpace>
//           FloatType
//       VectorType<ScalarType, VectorSize>

struct Type {
	static std::string Name() { return ".<unknown>"; }

	enum class Kind {
		Void,
		Bit,
		Int,
		UInt,
		Float,
		Vector,
		Array,
		Pointer
	};
	virtual Kind GetKind() const = 0;
	virtual Bits GetBits() const = 0;

	using SystemType = std::nullptr_t;
	using WideType = std::nullptr_t;

	using ComparisonOperator = std::nullptr_t;
	using ReductionOperation = std::nullptr_t;

	// Polymorphism
	virtual ~Type() = default;
};

struct VoidType : Type
{
	constexpr static Bits TypeBits = Bits::Bits1;

	using SystemType = bool;

	static std::string Name() { return ""; }

	Type::Kind GetKind() const override { return Kind::Void; }
	Bits GetBits() const override { return TypeBits; }
};

struct DataType : Type {};

struct ValueType : DataType {};

struct ScalarType : ValueType {};

// @struct is_type_specialization
//
// Type trait for determining if a type is a specialization of some template

template <class T, template <Bits, unsigned int> class Template>
struct is_type_specialization : std::false_type {};

template <template <Bits, unsigned int> class Template, Bits B, unsigned int N>
struct is_type_specialization<Template<B, N>, Template> : std::true_type {};
 
// @struct BitSize
//
// Convenience class for computing the number of bits or bytes in a type

template<Bits B>
struct BitSize
{
	constexpr static std::underlying_type<Bits>::type NumBits = static_cast<std::underlying_type<Bits>::type>(B);
	constexpr static std::underlying_type<Bits>::type NumBytes = NumBits / 8;
};

struct DynamicBitSize
{
	static std::underlying_type<Bits>::type GetBits(Bits B)
	{
		return static_cast<std::underlying_type<Bits>::type>(B);
	}
	static std::underlying_type<Bits>::type GetByte(Bits B)
	{
		return static_cast<std::underlying_type<Bits>::type>(B) / 8;
	}
};

// @struct BitType
//
// Untyped data representation

template<Bits B, unsigned int N = 1>
struct BitTypeBase : ScalarType
{
	constexpr static Bits TypeBits = B;

	static std::string Name() { return ".b" + std::to_string(BitSize<B>::NumBits); }

	Type::Kind GetKind() const override { return Kind::Bit; }
	Bits GetBits() const override { return TypeBits; }

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
		return ".<unknown>";
	}

	enum class AtomicOperation {
		And,
		Or,
		Xor,
		Exchange,
		CompareAndSwap
	};

	static std::string AtomicOperationString(AtomicOperation atomicOperation)
	{
		switch (atomicOperation)
		{
			case AtomicOperation::And:
				return ".and";
			case AtomicOperation::Or:
				return ".or";
			case AtomicOperation::Xor:
				return ".xor";
			case AtomicOperation::Exchange:
				return ".exch";
			case AtomicOperation::CompareAndSwap:
				return ".cas";
		}
		return ".<unknown>";
	}

	enum class ReductionOperation {
		And,
		Or,
		Xor
	};

	static std::string ReductionOperationString(ReductionOperation reductionOperation)
	{
		switch (reductionOperation)
		{
			case ReductionOperation::And:
				return ".and";
			case ReductionOperation::Or:
				return ".or";
			case ReductionOperation::Xor:
				return ".xor";
		}
		return ".<unknown>";
	}
};

template<>
struct BitTypeBase<Bits::Bits1, 1> : ScalarType
{
	constexpr static Bits TypeBits = Bits::Bits1;

	static std::string Name() { return ".pred"; }

	Type::Kind GetKind() const override { return Kind::Bit; }
	Bits GetBits() const override { return TypeBits; }
};

template<>
struct BitTypeBase<Bits::Bits8, 1> : ScalarType
{
	constexpr static Bits TypeBits = Bits::Bits8;

	static std::string Name() { return ".b" + std::to_string(BitSize<Bits::Bits8>::NumBits); }

	Kind GetKind() const override { return Kind::Bit; }
	Bits GetBits() const override { return TypeBits; }
};

template<Bits B, unsigned int N = 1> struct BitType : BitTypeBase<B, N> {};
template<> struct BitType<Bits::Bits1, 1> : BitTypeBase<Bits::Bits1>
{
	using SystemType = bool;
	static std::string TypePrefix() { return "p"; }
};
template<> struct BitType<Bits::Bits8, 1> : BitTypeBase<Bits::Bits8>
{
	using SystemType = int8_t;
	static std::string TypePrefix() { return "bc"; }
};
template<> struct BitType<Bits::Bits16, 1> : BitTypeBase<Bits::Bits16>
{
	using SystemType = int16_t;
	static std::string TypePrefix() { return "bs"; }
};
template<> struct BitType<Bits::Bits32, 1> : BitTypeBase<Bits::Bits32>
{
	using SystemType = int32_t;
	static std::string TypePrefix() { return "b"; }
};
template<> struct BitType<Bits::Bits64, 1> : BitTypeBase<Bits::Bits64>
{
	using SystemType = int64_t;
	static std::string TypePrefix() { return "bd"; }
};

using PredicateType = BitType<Bits::Bits1>;
using Bit8Type = BitType<Bits::Bits8>;
using Bit16Type = BitType<Bits::Bits16>;
using Bit32Type = BitType<Bits::Bits32>;
using Bit64Type = BitType<Bits::Bits64>;

// @struct is_bit_type
//
// Type trait for determining if a PTX::Type is a bit type

template <class T, typename E = void>
struct is_predicate_type : std::false_type {};

template <class T>
struct is_predicate_type<T, std::enable_if_t<
	std::is_same<T, PredicateType>::value
>> : std::true_type {};

template <class T, typename E = void>
struct is_bit_type : std::false_type {};

template <class T>
struct is_bit_type<T, std::enable_if_t<
	is_type_specialization<T, BitType>::value
>> : std::true_type {};

// @struct IntType
//
// Signed integer representation

template<Bits B, unsigned int N = 1>
struct IntTypeBase : BitType<B, N>
{
	static_assert(N == 1, "PTX::IntType expects data packing of 1");

	static std::string Name() { return ".s" + std::to_string(BitSize<B>::NumBits); }

	Type::Kind GetKind() const override { return Type::Kind::Int; }

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
		return ".<unknown>";
	}

	enum class AtomicOperation {
		Add,
		Increment,
		Decrement,
		Minimum,
		Maximum
	};

	static std::string AtomicOperationString(AtomicOperation atomicOperation)
	{
		switch (atomicOperation)
		{
			case AtomicOperation::Add:
				return ".add";
			case AtomicOperation::Increment:
				return ".inc";
			case AtomicOperation::Decrement:
				return ".dec";
			case AtomicOperation::Minimum:
				return ".min";
			case AtomicOperation::Maximum:
				return ".max";
		}
		return ".<unknown>";
	}

	using ReductionOperation = AtomicOperation;

	static std::string ReductionOperationString(ReductionOperation reductionOperation)
	{
		return AtomicOperationString(reductionOperation);
	}
};

template<>
struct IntTypeBase<Bits::Bits8, 1> : BitType<Bits::Bits8>
{
	static std::string Name() { return ".s" + std::to_string(BitSize<Bits::Bits8>::NumBits); }

	Type::Kind GetKind() const override { return Type::Kind::Int; }
};

template<Bits B, unsigned int N = 1> struct IntType : IntTypeBase<B, N> {};
template<> struct IntType<Bits::Bits8, 1> : IntTypeBase<Bits::Bits8>
{
	using SystemType = int8_t;
	static std::string TypePrefix() { return "rc"; }

	// Disable reduction
	using ReductionOperation = std::nullptr_t;
};
template<> struct IntType<Bits::Bits16, 1> : IntTypeBase<Bits::Bits16>
{
	using SystemType = int16_t;
	using WideType = IntType<Bits::Bits32>;
	static std::string TypePrefix() { return "rs"; }

	// Disable reduction
	using ReductionOperation = std::nullptr_t;
};
template<> struct IntType<Bits::Bits32, 1> : IntTypeBase<Bits::Bits32>
{
	using SystemType = int32_t;
	using WideType = IntType<Bits::Bits64>;
	static std::string TypePrefix() { return "r"; }
};
template<> struct IntType<Bits::Bits64, 1> : IntTypeBase<Bits::Bits64>
{
	using SystemType = int64_t;
	static std::string TypePrefix() { return "rd"; }
};

using Int8Type = IntType<Bits::Bits8>;
using Int16Type = IntType<Bits::Bits16>;
using Int32Type = IntType<Bits::Bits32>;
using Int64Type = IntType<Bits::Bits64>;

// @struct is_signed_int_type
//
// Type trait for determining if a PTX::Type is a signed integer type

template <class T, typename E = void>
struct is_signed_int_type : std::false_type {};

template <class T>
struct is_signed_int_type<T, std::enable_if_t<
	is_type_specialization<T, IntType>::value
>> : std::true_type {};

// @struct UIntType
//
// Unsigned integer representation

template<Bits B, unsigned int N = 1>
struct UIntTypeBase : BitType<B, N>
{
	static_assert(N == 1, "PTX::UIntType expects data packing of 1");

	static std::string Name() { return ".u" + std::to_string(BitSize<B>::NumBits); }

	Type::Kind GetKind() const override { return Type::Kind::UInt; }

	enum class ComparisonOperator {
		Equal,
		NotEqual,
		Less,
		LessEqual,
		Greater,
		GreaterEqual,
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
			case ComparisonOperator::Less:
				return ".lt";
			case ComparisonOperator::LessEqual:
				return ".le";
			case ComparisonOperator::Greater:
				return ".gt";
			case ComparisonOperator::GreaterEqual:
				return ".ge";
			case ComparisonOperator::Lower:
				return ".lo";
			case ComparisonOperator::LowerSame:
				return ".ls";
			case ComparisonOperator::Higher:
				return ".hi";
			case ComparisonOperator::HigherSame:
				return ".hs";
		}
		return ".<unknown>";
	}

	enum class AtomicOperation {
		Add,
		Increment,
		Decrement,
		Minimum,
		Maximum
	};

	static std::string AtomicOperationString(AtomicOperation atomicOperation)
	{
		switch (atomicOperation)
		{
			case AtomicOperation::Add:
				return ".add";
			case AtomicOperation::Increment:
				return ".inc";
			case AtomicOperation::Decrement:
				return ".dec";
			case AtomicOperation::Minimum:
				return ".min";
			case AtomicOperation::Maximum:
				return ".max";
		}
		return ".<unknown>";
	}

	using ReductionOperation = AtomicOperation;

	static std::string ReductionOperationString(ReductionOperation reductionOperation)
	{
		return AtomicOperationString(reductionOperation);
	}
};

template<>
struct UIntTypeBase<Bits::Bits8, 1> : BitType<Bits::Bits8>
{
	static std::string Name() { return ".u" + std::to_string(BitSize<Bits::Bits8>::NumBits); }
};

template<Bits B, unsigned int N = 1> struct UIntType : UIntTypeBase<B, N> {};
template<> struct UIntType<Bits::Bits8, 1> : UIntTypeBase<Bits::Bits8>
{
	using SystemType = uint8_t;
	static std::string TypePrefix() { return "uc"; }
};
template<> struct UIntType<Bits::Bits16, 1> : UIntTypeBase<Bits::Bits16>
{
	using SystemType = uint16_t;
	using WideType = UIntType<Bits::Bits32>;
	static std::string TypePrefix() { return "us"; }
};
template<> struct UIntType<Bits::Bits32, 1> : UIntTypeBase<Bits::Bits32>
{
	using SystemType = uint32_t;
	using WideType = UIntType<Bits::Bits64>;
	static std::string TypePrefix() { return "u"; }
};
template<> struct UIntType<Bits::Bits64, 1> : UIntTypeBase<Bits::Bits64>
{
	using SystemType = uint64_t;
	static std::string TypePrefix() { return "ud"; }
};

using UInt8Type = UIntType<Bits::Bits8>;
using UInt16Type = UIntType<Bits::Bits16>;
using UInt32Type = UIntType<Bits::Bits32>;
using UInt64Type = UIntType<Bits::Bits64>;

// @struct is_unsigned_int_type
//
// Type trait for determining if a PTX::Type is an unsigned integer type

template <class T, typename E = void>
struct is_unsigned_int_type : std::false_type {};

template <class T>
struct is_unsigned_int_type<T, std::enable_if_t<
	is_type_specialization<T, UIntType>::value
>> : std::true_type {};

// @struct is_int_type
//
// Type trait for determining if a PTX::Type is an integer (signed or unsigned) type

template <class T, typename E = void>
struct is_int_type : std::false_type {};

template <class T>
struct is_int_type<T, std::enable_if_t<
	is_type_specialization<T, IntType>::value ||
	is_type_specialization<T, UIntType>::value
>> : std::true_type {};

// @struct FloatType
//
// Floating point representation

template<Bits B, unsigned int N = 1>
struct FloatTypeBase : BitType<B, N>
{
	static_assert(N == 1, "PTX::FloatType expects data packing of 1");

	static std::string Name() { return ".f" + std::to_string(BitSize<B>::NumBits); }

	Type::Kind GetKind() const override { return Type::Kind::Float; }

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
			case RoundingMode::None:
				return "";
			case RoundingMode::Nearest:
				return ".rn";
			case RoundingMode::Zero:
				return ".rz";
			case RoundingMode::NegativeInfinity:
				return ".rm";
			case RoundingMode::PositiveInfinity:
				return ".rp";
		}
		return ".<unknown>";
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
		return ".<unknown>";
	}

	enum class AtomicOperation {
		Add,
		Increment,
		Decrement,
		Minimum,
		Maximum
	};

	static std::string AtomicOperationString(AtomicOperation atomicOperation)
	{
		switch (atomicOperation)
		{
			case AtomicOperation::Add:
				return ".add";
			case AtomicOperation::Increment:
				return ".inc";
			case AtomicOperation::Decrement:
				return ".dec";
			case AtomicOperation::Minimum:
				return ".min";
			case AtomicOperation::Maximum:
				return ".max";
		}
		return ".<unknown>";
	}

	using ReductionOperation = AtomicOperation;

	static std::string ReductionOperationString(ReductionOperation reductionOperation)
	{
		return AtomicOperationString(reductionOperation);
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

	Type::Kind GetKind() const override { return Type::Kind::Float; }

	enum class RoundingMode {
		None,
		Nearest,
	};

	static std::string RoundingModeString(RoundingMode roundingMode)
	{
		switch (roundingMode)
		{
			case RoundingMode::None:
				return "";
			case RoundingMode::Nearest:
				return ".rn";
		}
		return ".<unknown>";
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
		return ".<unknown>";
	}

	enum class AtomicOperation {
		Add
	};

	static std::string AtomicOperationString(AtomicOperation atomicOperation)
	{
		switch (atomicOperation)
		{
			case AtomicOperation::Add:
				return ".add.noftz";
		}
		return ".<unknown>";
	}

	using ReductionOperation = AtomicOperation;

	static std::string ReductionOperationString(ReductionOperation reductionOperation)
	{
		return AtomicOperationString(reductionOperation);
	}
};

template<Bits B, unsigned int N = 1> struct FloatType : FloatTypeBase<B, N> {};
template<> struct FloatType<Bits::Bits16, 1> : FloatTypeBase<Bits::Bits16>
{
	using SystemType = float;
	static std::string TypePrefix() { return "h"; }
};
template<> struct FloatType<Bits::Bits16, 2> : FloatTypeBase<Bits::Bits16>
{
	static std::string TypePrefix() { return "hh"; }
};
template<> struct FloatType<Bits::Bits32, 1> : FloatTypeBase<Bits::Bits32>
{
	using SystemType = float;
	static std::string TypePrefix() { return "f"; }
};
template<> struct FloatType<Bits::Bits64, 1> : FloatTypeBase<Bits::Bits64>
{
	using SystemType = double;
	static std::string TypePrefix() { return "fd"; }
};

using Float16Type = FloatType<Bits::Bits16>;
using Float16x2Type = FloatType<Bits::Bits16, 2>;
using Float32Type = FloatType<Bits::Bits32>;
using Float64Type = FloatType<Bits::Bits64>;

// @struct is_float_type
//
// Type trait for determining if a PTX::Type is a floating point type

template <class T, typename E = void>
struct is_float_type : std::false_type {};

template <class T>
struct is_float_type<T, std::enable_if_t<
	is_type_specialization<T, FloatType>::value
>> : std::true_type {};

enum class VectorSize : int {
	Vector2 = 2,
	Vector4 = 4
};

// @struct VectorType
//
// Type for representing pairs of data

template<VectorSize V>
struct VectorProperties
{
	constexpr static std::underlying_type<VectorSize>::type ElementCount = static_cast<std::underlying_type<VectorSize>::type>(V);
};

struct _VectorType : ValueType
{
	virtual VectorSize GetSize() const = 0;
	virtual const Type *GetType() const = 0;
};

template<class T, VectorSize V, bool Assert = true>
struct VectorType : _VectorType
{
	REQUIRE_TYPE_PARAM(VectorType, 
		REQUIRE_BASE(T, ScalarType)
	);

	using SystemType = typename T::SystemType;
	using ElementType = T;
	constexpr static Bits TypeBits = T::TypeBits;

	static std::string Name() { return ".v" + std::to_string(VectorProperties<V>::ElementCount) + " " + T::Name(); }

	Type::Kind GetKind() const override { return Type::Kind::Vector; }
	Bits GetBits() const override { return T::TypeBits; }

	VectorSize GetSize() const override { return V; }
	const Type *GetType() const override { return &m_type; }

	T m_type;
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

// @struct is_array_type
//
// Type trait for determining if a PTX::Type is an array type

template<class, unsigned int> struct ArrayType;

template <class T>
struct is_array_type : std::false_type {};

template <class T, unsigned int N>
struct is_array_type<ArrayType<T, N>> : std::true_type {};

const unsigned int DynamicSize = 0;

struct _ArrayType : DataType
{
	virtual unsigned int GetDimension() const = 0;
	virtual const Type *GetType() const = 0;
};

// @struct ArrayType
//
// Representation of array types

template<class T, unsigned int N>
struct ArrayType : _ArrayType
{
	REQUIRE_TYPE_PARAM(ArrayType,
		REQUIRE_BASE(T, DataType)
	);

	using SystemType = typename T::SystemType;
	using ElementType = T;
	constexpr static unsigned int ElementCount = N;
	constexpr static Bits TypeBits = T::TypeBits;

	static std::string TypePrefix() { return std::string("a$") + T::TypePrefix(); }

	static std::string BaseName()
	{
		if constexpr(is_array_type<T>::value)
		{
			return T::BaseName();
		}
		else
		{
			return T::Name();
		}
	}

	static std::string Dimensions()
	{
		std::string code = "[";
		if constexpr(N != DynamicSize)
		{
			code += std::to_string(N);
		}
		code += "]";
		if constexpr(is_array_type<T>::value)
		{
			code += T::Dimensions();
		}
		return code;
	}

	static std::string Name() { return BaseName() + Dimensions(); }

	Type::Kind GetKind() const override { return Type::Kind::Array; }
	Bits GetBits() const override { return TypeBits; }

	unsigned int GetDimension() const override { return N; }
	const Type *GetType() const override { return &m_type; }

	T m_type;
};

// @struct PointerType
//
// Representation of pointer (unsigned integer) types

template<Bits B>
struct _PointerType : UIntType<B>
{
	virtual const StateSpace *GetStateSpace() const = 0;
	virtual const Type *GetType() const = 0;
};

template<Bits B, class T, class S = AddressableSpace>
struct PointerType : _PointerType<B>
{
	REQUIRE_TYPE_PARAM(PointerType, 
		REQUIRE_BASE(T, ValueType)
	);
	REQUIRE_SPACE_PARAM(PointerType,
		REQUIRE_BASE(S, AddressableSpace)
	);

	static std::string Name() { return UIntType<B>::Name(); }

	Type::Kind GetKind() const override { return Type::Kind::Pointer; }

	const Type *GetType() const override { return &m_type; }
	const StateSpace *GetStateSpace() const override { return &m_space; }

	T m_type;
	S m_space;
};

template<class T, class S = AddressableSpace>
using Pointer32Type = PointerType<Bits::Bits32, T, S>;
template<class T, class S = AddressableSpace>
using Pointer64Type = PointerType<Bits::Bits64, T, S>;

// @struct is_pointer_type
//
// Type trait for determining if a PTX::Type is a pointer

template <class T, template <Bits, class, class> class Template>
struct is_pointer_specialization : std::false_type {};

template <template <Bits, class, class> class Template, Bits B, class T, class S>
struct is_pointer_specialization<Template<B, T, S>, Template> : std::true_type {};
 
template <class T, typename E = void>
struct is_pointer_type : std::false_type {};

template <class T>
struct is_pointer_type<T, std::enable_if_t<
	is_pointer_specialization<T, PointerType>::value
>> : std::true_type {};

// @struct is_comparable_type
//
// Type trait for determining if a PTX::Type can be compared

template<class T, typename E = void>
struct is_comparable_type : std::false_type {};

template<class T>
struct is_comparable_type<T, std::enable_if_t<std::is_enum<typename T::ComparisonOperator>::value>> : std::true_type {};

// @struct is_rounding_type
//
// Type trait for determining if a PTX::Type has a rounding mode

template <class T, typename E = void>
struct is_rounding_type : std::false_type {};

template <class T>
struct is_rounding_type<T, std::enable_if_t<std::is_enum<typename T::RoundingMode>::value>> : std::true_type {};

// @struct is_reduction_type
//
// Type trait for determining if a PTX::Type has reduction support

template <class T, typename E = void>
struct is_reduction_type : std::false_type {};

template <class T>
struct is_reduction_type<T, std::enable_if_t<std::is_enum<typename T::ReductionOperation>::value>> : std::true_type {};

}
