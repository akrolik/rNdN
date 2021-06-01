#pragma once

#include "Utils/Logger.h"

namespace PTX {

#define COMMA ,

#define DispatchSpace(space) \
	switch (space->GetKind()) { \
		case StateSpace::Kind::Register: \
			_DispatchSpace(space, RegisterSpace); break; \
		case StateSpace::Kind::SpecialRegister: \
			_DispatchSpace(space, SpecialRegisterSpace); break; \
		case StateSpace::Kind::Addressable: \
			_DispatchSpace(space, AddressableSpace); break; \
		case StateSpace::Kind::Local: \
			_DispatchSpace(space, LocalSpace); break; \
		case StateSpace::Kind::Global: \
			_DispatchSpace(space, GlobalSpace); break; \
		case StateSpace::Kind::Shared: \
			_DispatchSpace(space, SharedSpace); break; \
		case StateSpace::Kind::Const: \
			_DispatchSpace(space, ConstSpace); break; \
		case StateSpace::Kind::Parameter: \
			_DispatchSpace(space, ParameterSpace); break; \
	} \
	Utils::Logger::LogError("PTX::Dispatch unsupported space");

#define DispatchType_Void(type) \
	if (type->GetKind() == Type::Kind::Void) { \
		_DispatchType(type, VoidType); \
	}

#define DispatchType_Basic(type, f) \
	switch (type->GetKind()) { \
		case Type::Kind::Bit: { \
			switch (type->GetBits()) { \
				case Bits::Bits1: \
					_DispatchType(type, f(PredicateType)); break; \
				case Bits::Bits8: \
					_DispatchType(type, f(Bit8Type)); break; \
				case Bits::Bits16: \
					_DispatchType(type, f(Bit16Type)); break; \
				case Bits::Bits32: \
					_DispatchType(type, f(Bit32Type)); break; \
				case Bits::Bits64: \
					_DispatchType(type, f(Bit64Type)); break; \
			} \
			Utils::Logger::LogError("PTX::Dispatch unsupported type PTX::BitType"); \
		} \
		case Type::Kind::Int: { \
			switch (type->GetBits()) { \
				case Bits::Bits8: \
					_DispatchType(type, f(Int8Type)); break; \
				case Bits::Bits16: \
					_DispatchType(type, f(Int16Type)); break; \
				case Bits::Bits32: \
					_DispatchType(type, f(Int32Type)); break; \
				case Bits::Bits64: \
					_DispatchType(type, f(Int64Type)); break; \
			} \
			Utils::Logger::LogError("PTX::Dispatch unsupported type PTX::IntType"); \
		} \
		case Type::Kind::UInt: { \
			switch (type->GetBits()) { \
				case Bits::Bits8: \
					_DispatchType(type, f(UInt8Type)); break; \
				case Bits::Bits16: \
					_DispatchType(type, f(UInt16Type)); break; \
				case Bits::Bits32: \
					_DispatchType(type, f(UInt32Type)); break; \
				case Bits::Bits64: \
					_DispatchType(type, f(UInt64Type)); break; \
			} \
			Utils::Logger::LogError("PTX::Dispatch unsupported type PTX::UIntType"); \
		} \
		case Type::Kind::Float: { \
			switch (type->GetBits()) { \
				case Bits::Bits32: \
					_DispatchType(type, f(Float32Type)); break; \
				case Bits::Bits64: \
					_DispatchType(type, f(Float64Type)); break; \
			} \
			Utils::Logger::LogError("PTX::Dispatch unsupported type PTX::FloatType"); \
		} \
	}

#define DispatchExpand_Vector2(type) VectorType<type, VectorSize::Vector2>
#define DispatchExpand_Vector4(type) VectorType<type, VectorSize::Vector4>

#define DispatchType_Vector(type) \
	if (type->GetKind() == Type::Kind::Vector) { \
		const auto vtype = static_cast<const _VectorType *>(type); \
		switch (vtype->GetSize()) { \
			case VectorSize::Vector2: \
				DispatchType_Basic(vtype->GetType(), DispatchExpand_Vector2); break; \
			case VectorSize::Vector4: \
				DispatchType_Basic(vtype->GetType(), DispatchExpand_Vector4); break; \
		} \
		Utils::Logger::LogError("PTX::Dispatch unsupported type PTX::VectorType"); \
	}

#define DispatchExpand_PointerAddressable(type) PointerType<Bits::Bits64, type, AddressableSpace>
#define DispatchExpand_PointerLocal(type) PointerType<Bits::Bits64, type, LocalSpace>
#define DispatchExpand_PointerGlobal(type) PointerType<Bits::Bits64, type, GlobalSpace>
#define DispatchExpand_PointerShared(type) PointerType<Bits::Bits64, type, SharedSpace>
#define DispatchExpand_PointerConst(type) PointerType<Bits::Bits64, type, ConstSpace>
#define DispatchExpand_PointerParameter(type) PointerType<Bits::Bits64, type, ParameterSpace>

#define DispatchExpand_PointerGlobal2(type) PointerType<Bits::Bits64, PointerType<Bits::Bits64, type, GlobalSpace>>

#define DispatchType_Pointer(type) \
	if (type->GetKind() == Type::Kind::Pointer) { \
		if (type->GetBits() == Bits::Bits64) { \
			const auto ptype = static_cast<const _PointerType<Bits::Bits64> *>(type); \
			const auto vtype = static_cast<const _PointerType<Bits::Bits64> *>(ptype->GetType()); \
			switch (ptype->GetStateSpace()->GetKind()) { \
				case StateSpace::Kind::Addressable: \
					if (vtype->GetKind() == Type::Kind::Pointer) { \
						if (vtype->GetBits() == Bits::Bits64) { \
							const auto ptype2 = static_cast<const _PointerType<Bits::Bits64> *>(vtype); \
							if (ptype2->GetStateSpace()->GetKind() == StateSpace::Kind::Global) { \
									DispatchType_Basic(ptype2->GetType(), DispatchExpand_PointerGlobal2); \
							} \
						} \
					} else { \
						DispatchType_Basic(vtype, DispatchExpand_PointerAddressable); \
					} \
					break; \
				case StateSpace::Kind::Local: \
					DispatchType_Basic(vtype, DispatchExpand_PointerLocal); break; \
				case StateSpace::Kind::Global: \
					DispatchType_Basic(vtype, DispatchExpand_PointerGlobal); break; \
				case StateSpace::Kind::Shared: \
					DispatchType_Basic(vtype, DispatchExpand_PointerShared); break; \
				case StateSpace::Kind::Const: \
					DispatchType_Basic(vtype, DispatchExpand_PointerConst); break; \
				case StateSpace::Kind::Parameter: \
					DispatchType_Basic(vtype, DispatchExpand_PointerParameter); break; \
			} \
		} \
		Utils::Logger::LogError("PTX::Dispatch unsupported type PTX::PointerType"); \
	}

#define DispatchExpand_Array32(type) ArrayType<type, 32>
#define DispatchExpand_Array64(type) ArrayType<type, 64>
#define DispatchExpand_Array128(type) ArrayType<type, 128>
#define DispatchExpand_Array256(type) ArrayType<type, 256>
#define DispatchExpand_Array512(type) ArrayType<type, 512>
#define DispatchExpand_Array1024(type) ArrayType<type, 1024>
#define DispatchExpand_Array2048(type) ArrayType<type, 2048>

#define DispatchType_Array(type) \
	if (type->GetKind() == Type::Kind::Array) { \
		const auto atype = static_cast<const _ArrayType *>(type); \
		switch (atype->GetDimension()) { \
			case 32: \
				DispatchType_Basic(atype->GetType(), DispatchExpand_Array32); break; \
			case 64: \
				DispatchType_Basic(atype->GetType(), DispatchExpand_Array64); break; \
			case 128: \
				DispatchType_Basic(atype->GetType(), DispatchExpand_Array128); break; \
			case 256: \
				DispatchType_Basic(atype->GetType(), DispatchExpand_Array256); break; \
			case 512: \
				DispatchType_Basic(atype->GetType(), DispatchExpand_Array512); break; \
			case 1024: \
				DispatchType_Basic(atype->GetType(), DispatchExpand_Array1024); break; \
			case 2048: \
				DispatchType_Basic(atype->GetType(), DispatchExpand_Array2048); break; \
		} \
		Utils::Logger::LogError("PTX::Dispatch unsupported type PTX::ArrayType"); \
	}

#define DispatchExpand_Null(type) type

#define DispatchType(type) \
	DispatchType_Basic(type, DispatchExpand_Null); \
	DispatchType_Vector(type); \
	DispatchType_Pointer(type); \
	DispatchType_Array(type); \
	DispatchType_Void(type); \
	Utils::Logger::LogError("PTX::Dispatch unsupported type");

template<template<class, bool = true> class C>
class Dispatcher
{
public:
	template<class V>
	void Dispatch(V& visitor) const
	{
#define _DispatchType(type, T) \
		if constexpr(C<T, false>::TypeSupported) { \
			return visitor.Visit(static_cast<const C<T>*>(this)); \
		}

		const auto type = GetType();
		DispatchType(type);

#undef _DispatchType
	}

	template<class V>
	void Dispatch(V& visitor)
	{
#define _DispatchType(type, T) \
		if constexpr(C<T, false>::TypeSupported) { \
			return visitor.Visit(static_cast<C<T>*>(this)); \
		}

		const auto type = GetType();
		DispatchType(type);

#undef _DispatchType
	}
protected:
	virtual const Type *GetType() const = 0;
};

template<template<class, class, bool = true> class C>
class Dispatcher_2
{
public:
	template<class V>
	void Dispatch(V& visitor) const
	{
#define _DispatchType(type, T1) \
		return Dispatch<V, T1>(visitor);

		const auto type = GetType1();
		DispatchType(type);

#undef _DispatchType
	}

	template<class V, class T1>
	void Dispatch(V& visitor) const
	{
#define _DispatchType(type, T2) \
		if constexpr(C<T1, T2, false>::TypeSupported) { \
			return visitor.Visit(static_cast<const C<T1, T2>*>(this)); \
		}

		const auto type = GetType2();
		DispatchType(type);

#undef _DispatchType
	}

	template<class V>
	void Dispatch(V& visitor)
	{
#define _DispatchType(type, T1) \
		return Dispatch<V, T1>(visitor);

		const auto type = GetType1();
		DispatchType(type);

#undef _DispatchType
	}

	template<class V, class T1>
	void Dispatch(V& visitor)
	{
#define _DispatchType(type, T2) \
		if constexpr(C<T1, T2, false>::TypeSupported) { \
			return visitor.Visit(static_cast<C<T1, T2>*>(this)); \
		}

		const auto type = GetType2();
		DispatchType(type);

#undef _DispatchType
	}

protected:
	virtual const Type *GetType1() const = 0;
	virtual const Type *GetType2() const = 0;
};

template<template<class, VectorSize, bool = true> class C>
class Dispatcher_Vector
{
public:
	template<class V>
	void Dispatch(V& visitor) const
	{
		const auto vectorSize = GetVectorSize();
		if (vectorSize == VectorSize::Vector2)
		{
			Dispatch<V, VectorSize::Vector2>(visitor);
		}
		else if (vectorSize == VectorSize::Vector4)
		{
			Dispatch<V, VectorSize::Vector4>(visitor);
		}
	}

	template<class V, VectorSize E>
	void Dispatch(V& visitor) const
	{
#define _DispatchType(type, T) \
		if constexpr(C<T, E, false>::TypeSupported) { \
			return visitor.Visit(static_cast<const C<T, E>*>(this)); \
		}

		const auto type = GetType();
		DispatchType(type)

#undef _DispatchType
	}

	template<class V>
	void Dispatch(V& visitor)
	{
		const auto vectorSize = GetVectorSize();
		if (vectorSize == VectorSize::Vector2)
		{
			Dispatch<V, VectorSize::Vector2>(visitor);
		}
		else if (vectorSize == VectorSize::Vector4)
		{
			Dispatch<V, VectorSize::Vector4>(visitor);
		}
	}

	template<class V, VectorSize E>
	void Dispatch(V& visitor)
	{
#define _DispatchType(type, T) \
		if constexpr(C<T, E, false>::TypeSupported) { \
			return visitor.Visit(static_cast<C<T, E>*>(this)); \
		}

		const auto type = GetType();
		DispatchType(type)

#undef _DispatchType
	}

protected:
	virtual const Type *GetType() const = 0;
	virtual const VectorSize GetVectorSize() const = 0;
};

template<template<class, class, VectorSize, bool = true> class C>
class Dispatcher_VectorSpace
{
public:
	template<class V>
	void Dispatch(V& visitor) const
	{
		const auto vectorSize = GetVectorSize();
		if (vectorSize == VectorSize::Vector2)
		{
			Dispatch<V, VectorSize::Vector2>(visitor);
		}
		else if (vectorSize == VectorSize::Vector4)
		{
			Dispatch<V, VectorSize::Vector4>(visitor);
		}
	}

	template<class V, VectorSize E>
	void Dispatch(V& visitor) const
	{
#define _DispatchSpace(space, S) \
		return Dispatch<V, S, E>(visitor);

		const auto space = GetStateSpace();
		DispatchSpace(space);

#undef _DispatchSpace
	}

	template<class V, class S, VectorSize E>
	void Dispatch(V& visitor) const
	{
#define _DispatchType(type, T) \
		if constexpr(C<T, S, E, false>::TypeSupported && C<T, S, E, false>::SpaceSupported) { \
			return visitor.Visit(static_cast<const C<T, S, E>*>(this)); \
		}

		const auto type = GetType();
		DispatchType(type)

#undef _DispatchType
	}

	template<class V>
	void Dispatch(V& visitor)
	{
		const auto vectorSize = GetVectorSize();
		if (vectorSize == VectorSize::Vector2)
		{
			Dispatch<V, VectorSize::Vector2>(visitor);
		}
		else if (vectorSize == VectorSize::Vector4)
		{
			Dispatch<V, VectorSize::Vector4>(visitor);
		}
	}

	template<class V, VectorSize E>
	void Dispatch(V& visitor)
	{
#define _DispatchSpace(space, S) \
		return Dispatch<V, S, E>(visitor);

		const auto space = GetStateSpace();
		DispatchSpace(space);

#undef _DispatchSpace
	}

	template<class V, class S, VectorSize E>
	void Dispatch(V& visitor)
	{
#define _DispatchType(type, T) \
		if constexpr(C<T, S, E, false>::TypeSupported && C<T, S, E, false>::SpaceSupported) { \
			return visitor.Visit(static_cast<C<T, S, E>*>(this)); \
		}

		const auto type = GetType();
		DispatchType(type)

#undef _DispatchType
	}

protected:
	virtual const Type *GetType() const = 0;
	virtual const StateSpace *GetStateSpace() const = 0;
	virtual const VectorSize GetVectorSize() const = 0;
};

template<template<class, class, bool = true> class C>
class Dispatcher_Space
{
public:
	template<class V>
	void Dispatch(V& visitor) const
	{
#define _DispatchSpace(space, S) \
		return Dispatch<V, S>(visitor);

		const auto space = GetStateSpace();
		DispatchSpace(space);

#undef _DispatchSpace
	}

	template<class V, class S>
	void Dispatch(V& visitor) const
	{
#define _DispatchType(type, T) \
		if constexpr(C<T, S, false>::TypeSupported && C<T, S, false>::SpaceSupported) { \
			return visitor.Visit(static_cast<const C<T, S>*>(this)); \
		}

		const auto type = GetType();
		DispatchType(type);

#undef _DispatchType
	}

	template<class V>
	void Dispatch(V& visitor)
	{
#define _DispatchSpace(space, S) \
		return Dispatch<V, S>(visitor);

		const auto space = GetStateSpace();
		DispatchSpace(space);

#undef _DispatchSpace
	}

	template<class V, class S>
	void Dispatch(V& visitor)
	{
#define _DispatchType(type, T) \
		if constexpr(C<T, S, false>::TypeSupported && C<T, S, false>::SpaceSupported) { \
			return visitor.Visit(static_cast<C<T, S>*>(this)); \
		}

		const auto type = GetType();
		DispatchType(type);

#undef _DispatchType
	}

protected:
	virtual const Type *GetType() const = 0;
	virtual const StateSpace *GetStateSpace() const = 0;
};

template<template<Bits, class, class, bool = true> class C>
class Dispatcher_Data
{
public:
	template<class V>
	void Dispatch(V& visitor) const
	{
		const auto bits = GetBits();
		if (bits == Bits::Bits32)
		{
			Dispatch<V, Bits::Bits32>(visitor);
		}
		else if (bits == Bits::Bits64)
		{
			Dispatch<V, Bits::Bits64>(visitor);
		}
	}

	template<class V, Bits B>
	void Dispatch(V& visitor) const
	{
#define _DispatchSpace(space, S) \
		return Dispatch<V, B, S>(visitor);

		const auto space = GetStateSpace();
		DispatchSpace(space);

#undef _DispatchSpace
	}

	template<class V, Bits B, class S>
	void Dispatch(V& visitor) const
	{
#define _DispatchType(type, T) \
		if constexpr(C<B, T, S, false>::TypeSupported && C<B, T, S, false>::SpaceSupported) { \
			return visitor.Visit(static_cast<const C<B, T, S>*>(this)); \
		}

		const auto type = GetType();
		DispatchType(type);

#undef _DispatchType
	}

	template<class V>
	void Dispatch(V& visitor)
	{
		const auto bits = GetBits();
		if (bits == Bits::Bits32)
		{
			Dispatch<V, Bits::Bits32>(visitor);
		}
		else if (bits == Bits::Bits64)
		{
			Dispatch<V, Bits::Bits64>(visitor);
		}
	}

	template<class V, Bits B>
	void Dispatch(V& visitor)
	{
#define _DispatchSpace(space, S) \
		return Dispatch<V, B, S>(visitor);

		const auto space = GetStateSpace();
		DispatchSpace(space);

#undef _DispatchSpace
	}

	template<class V, Bits B, class S>
	void Dispatch(V& visitor)
	{
#define _DispatchType(type, T) \
		if constexpr(C<B, T, S, false>::TypeSupported && C<B, T, S, false>::SpaceSupported) { \
			return visitor.Visit(static_cast<C<B, T, S>*>(this)); \
		}

		const auto type = GetType();
		DispatchType(type);

#undef _DispatchType
	}

protected:
	virtual const Bits GetBits() const = 0;
	virtual const Type *GetType() const = 0;
	virtual const StateSpace *GetStateSpace() const = 0;
};

template<typename A, template<Bits, class, class, A, bool = true> class C>
class Dispatcher_DataAtomic
{
public:
	template<class V, A AA>
	void Dispatch(V& visitor) const
	{
		const auto bits = GetBits();
		if (bits == Bits::Bits32)
		{
			Dispatch<V, Bits::Bits32, AA>(visitor);
		}
		else if (bits == Bits::Bits64)
		{
			Dispatch<V, Bits::Bits64, AA>(visitor);
		}
	}

	template<class V, Bits B, A AA>
	void Dispatch(V& visitor) const
	{
#define _DispatchSpace(space, S) \
		return Dispatch<V, B, S, AA>(visitor);

		const auto space = GetStateSpace();
		DispatchSpace(space);

#undef _DispatchSpace
	}

	template<class V, Bits B, class S, A AA>
	void Dispatch(V& visitor) const
	{
#define _DispatchType(type, T) \
		if constexpr(C<B, T, S, AA, false>::TypeSupported && C<B, T, S, AA, false>::SpaceSupported) { \
			return visitor.Visit(static_cast<const C<B, T, S, AA>*>(this)); \
		}

		const auto type = GetType();
		DispatchType(type);

#undef _DispatchType
	}

	template<class V, A AA>
	void Dispatch(V& visitor)
	{
		const auto bits = GetBits();
		if (bits == Bits::Bits32)
		{
			Dispatch<V, Bits::Bits32, AA>(visitor);
		}
		else if (bits == Bits::Bits64)
		{
			Dispatch<V, Bits::Bits64, AA>(visitor);
		}
	}

	template<class V, Bits B, A AA>
	void Dispatch(V& visitor)
	{
#define _DispatchSpace(space, S) \
		return Dispatch<V, B, S, AA>(visitor);

		const auto space = GetStateSpace();
		DispatchSpace(space);

#undef _DispatchSpace
	}

	template<class V, Bits B, class S, A AA>
	void Dispatch(V& visitor)
	{
#define _DispatchType(type, T) \
		if constexpr(C<B, T, S, AA, false>::TypeSupported && C<B, T, S, AA, false>::SpaceSupported) { \
			return visitor.Visit(static_cast<C<B, T, S, AA>*>(this)); \
		}

		const auto type = GetType();
		DispatchType(type);

#undef _DispatchType
	}

protected:
	virtual const Bits GetBits() const = 0;
	virtual const Type *GetType() const = 0;
	virtual const StateSpace *GetStateSpace() const = 0;
	virtual const A GetAtomic() const = 0;
};

#define DispatchInherit(x) public _##x

#define DispatchInterface_Using(x) \
	class _##x : public Dispatcher<x> {};

#define DispatchInterface(x) \
	template<class T, bool Assert> class x; \
	class _##x : public Dispatcher<x> {};

#define DispatchInterface_2(x) \
	template<class T1, class T2, bool Assert> class x; \
	class _##x : public Dispatcher_2<x> {};

#define DispatchInterface_Vector(x) \
	template<class T, VectorSize V, bool Assert> class x; \
	class _##x : public Dispatcher_Vector<x> {};

#define DispatchInterface_VectorSpace(x) \
	template<class T, class S, VectorSize V, bool Assert> class x; \
	class _##x : public Dispatcher_VectorSpace<x> {};

#define DispatchInterface_Space(x) \
	template<class T, class S, bool Assert> class x; \
	class _##x : public Dispatcher_Space<x> {};
 
#define DispatchInterface_Data(x) \
	template<Bits B, class T, class S, bool Assert> class x; \
	class _##x : public Dispatcher_Data<x> {};

#define DispatchInterface_DataAtomic(x, y) \
	template<Bits B, class T, class S, y A, bool Assert> class x; \
	class _##x : public Dispatcher_DataAtomic<y, x> { \
	public: \
		template<class V> void Dispatch(V& visitor) const; \
		template<class V> void Dispatch(V& visitor); \
	};

#define DispatchImplementation_DataAtomic(x, y) \
	template<class V> \
	void _##x::Dispatch(V& visitor) const \
	{ \
		y; \
	} \
	template<class V> \
	void _##x::Dispatch(V& visitor) \
	{ \
		y; \
	}


#define DispatchMember_Bits(b) \
	const Bits GetBits() const override { return b; }
#define DispatchMember_Type(t) \
	const Type *GetType() const override { return &m_type; } \
	t m_type;
#define DispatchMember_Type1(x) \
	const Type *GetType1() const override { return &m_type1; } \
	x m_type1;
#define DispatchMember_Type2(x) \
	const Type *GetType2() const override { return &m_type2; } \
	x m_type2;
#define DispatchMember_Space(s) \
	const StateSpace *GetStateSpace() const override { return &m_space; } \
	s m_space;
#define DispatchMember_Vector(v) \
	const VectorSize GetVectorSize() const override { return v; }
#define DispatchMember_Atomic(t,v) \
	const t GetAtomic() const override { return v; }

#undef DispatchSpace
#undef DispatchType

}
