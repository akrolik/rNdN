#pragma once

namespace PTX {

#define DispatchSpace(space) \
	_DispatchSpace(space, RegisterSpace); \
	_DispatchSpace(space, LocalSpace); \
	_DispatchSpace(space, GlobalSpace); \
	_DispatchSpace(space, SharedSpace); \
	_DispatchSpace(space, ConstSpace); \
	_DispatchSpace(space, ParameterSpace); \
	_DispatchSpace(space, AddressableSpace); \

#define COMMA ,
#define DispatchPointer_Bits(type, T, S) \
	_DispatchType(type, Pointer32Type<T COMMA S>); \
	_DispatchType(type, Pointer64Type<T COMMA S>);

#define DispatchPointer_Space(type, T) \
	DispatchPointer_Bits(type, T, AddressableSpace); \
	DispatchPointer_Bits(type, T, LocalSpace); \
	DispatchPointer_Bits(type, T, GlobalSpace); \
	DispatchPointer_Bits(type, T, SharedSpace); \
	DispatchPointer_Bits(type, T, ConstSpace); \
	DispatchPointer_Bits(type, T, ParameterSpace);

#define DispatchPointer(type) \
	DispatchPointer_Space(type, Int8Type); \
	DispatchPointer_Space(type, Int16Type); \
	DispatchPointer_Space(type, Int32Type); \
	DispatchPointer_Space(type, Int64Type); \
	DispatchPointer_Space(type, UInt8Type); \
	DispatchPointer_Space(type, UInt16Type); \
	DispatchPointer_Space(type, UInt32Type); \
	DispatchPointer_Space(type, UInt64Type); \
	DispatchPointer_Space(type, Float16Type); \
	DispatchPointer_Space(type, Float16x2Type); \
	DispatchPointer_Space(type, Float32Type); \
	DispatchPointer_Space(type, Float64Type); \
	DispatchPointer_Space(type, PredicateType); \
	DispatchPointer_Space(type, Bit8Type); \
	DispatchPointer_Space(type, Bit16Type); \
	DispatchPointer_Space(type, Bit32Type); \
	DispatchPointer_Space(type, Bit64Type);

#define DispatchType(type) \
	DispatchPointer(type); \
	_DispatchType(type, VoidType); \
	_DispatchType(type, Int8Type); \
	_DispatchType(type, Int16Type); \
	_DispatchType(type, Int32Type); \
	_DispatchType(type, Int64Type); \
	_DispatchType(type, UInt8Type); \
	_DispatchType(type, UInt16Type); \
	_DispatchType(type, UInt32Type); \
	_DispatchType(type, UInt64Type); \
	_DispatchType(type, Float16Type); \
	_DispatchType(type, Float16x2Type); \
	_DispatchType(type, Float32Type); \
	_DispatchType(type, Float64Type); \
	_DispatchType(type, PredicateType); \
	_DispatchType(type, Bit8Type); \
	_DispatchType(type, Bit16Type); \
	_DispatchType(type, Bit32Type); \
	_DispatchType(type, Bit64Type);

template<template<class, bool = true> class C>
class Dispatcher
{
public:
	template<class V>
	void Dispatch(V& visitor) const
	{
#define _DispatchType(type, T) \
		if constexpr(C<T, false>::TypeSupported) { \
			if (dynamic_cast<const T*>(type)) { \
				return visitor.Visit(static_cast<const C<T>*>(this)); \
			} \
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
		if (dynamic_cast<const T1*>(type)) { \
			return Dispatch<V, T1>(visitor); \
		}

		const auto type = GetType1();
		DispatchType(type);

#undef _DispatchType
	}

	template<class V, class T1>
	void Dispatch(V& visitor) const
	{
#define _DispatchType(type, T2) \
		if constexpr(C<T1, T2, false>::TypeSupported) { \
			if (dynamic_cast<const T2*>(type)) { \
				return visitor.Visit(static_cast<const C<T1, T2>*>(this)); \
			} \
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
			if (dynamic_cast<const T*>(type)) { \
				return visitor.Visit(static_cast<const C<T, E>*>(this)); \
			} \
		}

		const auto type = GetType();
		DispatchType(type)

#undef _DispatchType
	}

protected:
	virtual const Type *GetType() const = 0;
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
		if (dynamic_cast<const S*>(space)) { \
			return Dispatch<V, S>(visitor); \
		}

		const auto space = GetStateSpace();
		DispatchSpace(space);

#undef _DispatchSpace
	}

	template<class V, class S>
	void Dispatch(V& visitor) const
	{
#define _DispatchType(type, T) \
		if constexpr(C<T, S, false>::TypeSupported && C<T, S, false>::SpaceSupported) { \
			if (dynamic_cast<const T*>(type)) { \
				return visitor.Visit(static_cast<const C<T, S>*>(this)); \
			} \
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
		if (dynamic_cast<const S*>(space)) { \
			return Dispatch<V, B, S>(visitor); \
		}

		const auto space = GetStateSpace();
		DispatchSpace(space);

#undef _DispatchSpace
	}

	template<class V, Bits B, class S>
	void Dispatch(V& visitor) const
	{
#define _DispatchType(type, T) \
		if constexpr(C<B, T, S, false>::TypeSupported && C<B, T, S, false>::SpaceSupported) { \
			if (dynamic_cast<const T*>(type)) { \
				return visitor.Visit(static_cast<const C<B, T, S>*>(this)); \
			} \
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

template<template<Bits, class, class, class, bool = true> class C>
class Dispatcher_Data2
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
#define _DispatchSpace(space, S1) \
		if (dynamic_cast<const S1*>(space)) { \
			return Dispatch<V, B, S1>(visitor); \
		}

		const auto space = GetStateSpace1();
		DispatchSpace(space);

#undef _DispatchSpace
	}

	template<class V, Bits B, class S1>
	void Dispatch(V& visitor) const
	{
#define _DispatchSpace(space, S2) \
		if (dynamic_cast<const S2*>(space)) { \
			Dispatch<V, B, S1, S2>(visitor); \
		}

		const auto space = GetStateSpace2();
		DispatchSpace(space);

#undef _DispatchSpace
	}

	template<class V, Bits B, class S1, class S2>
	void Dispatch(V& visitor) const
	{
#define _DispatchType(type, T) \
		if constexpr(C<B, T, S1, S2, false>::TypeSupported && C<B, T, S1, S2, false>::SpaceSupported) { \
			if (dynamic_cast<const T*>(type)) { \
				return visitor.Visit(static_cast<const C<B, T, S1, S2>*>(this)); \
			} \
		}

		const auto type = GetType();
		DispatchType(type);

#undef _DispatchType
	}

protected:
	virtual const Bits GetBits() const = 0;
	virtual const Type *GetType() const = 0;
	virtual const StateSpace *GetStateSpace1() const = 0;
	virtual const StateSpace *GetStateSpace2() const = 0;
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
		if (dynamic_cast<const S*>(space)) { \
			return Dispatch<V, B, S, AA>(visitor); \
		}

		const auto space = GetStateSpace();
		DispatchSpace(space);

#undef _DispatchSpace
	}

	template<class V, Bits B, class S, A AA>
	void Dispatch(V& visitor) const
	{
#define _DispatchType(type, T) \
		if constexpr(C<B, T, S, AA, false>::TypeSupported && C<B, T, S, AA, false>::SpaceSupported) { \
			if (dynamic_cast<const T*>(type)) { \
				return visitor.Visit(static_cast<const C<B, T, S, AA>*>(this)); \
			} \
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

#define DispatchInterface_Space(x) \
	template<class T, class S, bool Assert> class x; \
	class _##x : public Dispatcher_Space<x> {};
 
#define DispatchInterface_Data(x) \
	template<Bits B, class T, class S, bool Assert> class x; \
	class _##x : public Dispatcher_Data<x> {};

#define DispatchInterface_Data2(x) \
	template<Bits B, class T, class S1, class S2, bool Assert> class x; \
	class _##x : public Dispatcher_Data2<x> {};

#define DispatchInterface_DataAtomic(x, y) \
	template<Bits B, class T, class S, y A, bool Assert> class x; \
	class _##x : public Dispatcher_DataAtomic<y, x> { \
	public: \
		template<class V> void Dispatch(V& visitor) const; \
	};

#define DispatchImplementation_DataAtomic(x, y) \
	template<class V> \
	void _##x::Dispatch(V& visitor) const \
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
#define DispatchMember_Space1(s) \
	const StateSpace *GetStateSpace1() const override { return &m_space1; } \
	s m_space1;
#define DispatchMember_Space2(s) \
	const StateSpace *GetStateSpace2() const override { return &m_space2; } \
	s m_space2;
#define DispatchMember_Vector(v) \
	const VectorSize GetVectorSize() const override { return v; }
#define DispatchMember_Atomic(t,v) \
	const t GetAtomic() const override { return v; }

#undef DispatchSpace
#undef DispatchType

}
