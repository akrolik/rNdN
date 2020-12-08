#pragma once

namespace PTX {

#define InstructionDispatchSpace(space) \
	_InstructionDispatchSpace(space, RegisterSpace); \
	_InstructionDispatchSpace(space, LocalSpace); \
	_InstructionDispatchSpace(space, GlobalSpace); \
	_InstructionDispatchSpace(space, SharedSpace); \
	_InstructionDispatchSpace(space, ConstSpace); \
	_InstructionDispatchSpace(space, ParameterSpace);

#define InstructionDispatchType(type) \
	_InstructionDispatchType(type, VoidType); \
	_InstructionDispatchType(type, Int8Type); \
	_InstructionDispatchType(type, Int16Type); \
	_InstructionDispatchType(type, Int32Type); \
	_InstructionDispatchType(type, Int64Type); \
	_InstructionDispatchType(type, UInt8Type); \
	_InstructionDispatchType(type, UInt16Type); \
	_InstructionDispatchType(type, UInt32Type); \
	_InstructionDispatchType(type, UInt64Type); \
	_InstructionDispatchType(type, Float16Type); \
	_InstructionDispatchType(type, Float16x2Type); \
	_InstructionDispatchType(type, Float32Type); \
	_InstructionDispatchType(type, Float64Type); \
	_InstructionDispatchType(type, PredicateType); \
	_InstructionDispatchType(type, Bit8Type); \
	_InstructionDispatchType(type, Bit16Type); \
	_InstructionDispatchType(type, Bit32Type); \
	_InstructionDispatchType(type, Bit64Type);

class InstructionDispatch
{
public:
	template<class V, template<class, bool = true> class I>
	void Dispatch(V& visitor) const
	{
#define _InstructionDispatchType(type, T) \
		if constexpr(I<T, false>::TypeSupported) { \
			if (dynamic_cast<const T*>(type)) { \
				visitor.Visit(static_cast<const I<T>*>(this)); \
			} \
		}

		const auto type = GetType();
		InstructionDispatchType(type);

#undef _InstructionDispatchType
	}

protected:
	virtual const Type *GetType() const = 0;
};

class InstructionDispatch_2
{
public:
	template<class V, template<class, class, bool = true> class I>
	void Dispatch(V& visitor) const
	{
#define _InstructionDispatchType(type, T1) \
		if (dynamic_cast<const T1*>(type)) { \
			Dispatch<V, I, T1>(visitor); \
		}

		const auto type = GetType1();
		InstructionDispatchType(type);

#undef _InstructionDispatchType
	}

	template<class V, template<class, class, bool = true> class I, class T1>
	void Dispatch(V& visitor) const
	{
#define _InstructionDispatchType(type, T2) \
		if constexpr(I<T1, T2, false>::TypeSupported) { \
			if (dynamic_cast<const T2*>(type)) { \
				visitor.Visit(static_cast<const I<T1, T2>*>(this)); \
			} \
		}

		const auto type = GetType2();
		InstructionDispatchType(type);

#undef _InstructionDispatchType
	}

protected:
	virtual const Type *GetType1() const = 0;
	virtual const Type *GetType2() const = 0;
};

class InstructionDispatch_Vector
{
public:
	template<class V, template<class, VectorSize, bool = true> class I>
	void Dispatch(V& visitor) const
	{
		const auto vectorSize = GetVectorSize();
		if (vectorSize == VectorSize::Vector2)
		{
			Dispatch<V, I, VectorSize::Vector2>(visitor);
		}
		else if (vectorSize == VectorSize::Vector4)
		{
			Dispatch<V, I, VectorSize::Vector4>(visitor);
		}
	}

	template<class V, template<class, VectorSize, bool = true> class I, VectorSize E>
	void Dispatch(V& visitor) const
	{
#define _InstructionDispatchType(type, T) \
		if constexpr(I<T, E, false>::TypeSupported) { \
			if (dynamic_cast<const T*>(type)) { \
				visitor.Visit(static_cast<const I<T, E>*>(this)); \
			} \
		}

		const auto type = GetType();
		InstructionDispatchType(type)

#undef _InstructionDispatchType
	}

protected:
	virtual const Type *GetType() const = 0;
	virtual const VectorSize GetVectorSize() const = 0;
};

class InstructionDispatch_Data
{
public:
	template<class V, template<Bits, class, class, bool = true> class I>
	void Dispatch(V& visitor) const
	{
		const auto bits = GetBits();
		if (bits == Bits::Bits32)
		{
			Dispatch<V, I, Bits::Bits32>(visitor);
		}
		else if (bits == Bits::Bits64)
		{
			Dispatch<V, I, Bits::Bits64>(visitor);
		}
	}

	template<class V, template<Bits, class, class, bool = true> class I, Bits B>
	void Dispatch(V& visitor) const
	{
#define _InstructionDispatchSpace(space, S) \
		if (dynamic_cast<const S*>(space)) { \
			Dispatch<V, I, B, S>(visitor); \
		}

		const auto space = GetStateSpace();
		InstructionDispatchSpace(space);

#undef _InstructionDispatchSpace
	}

	template<class V, template<Bits, class, class, bool = true> class I, Bits B, class S>
	void Dispatch(V& visitor) const
	{
#define _InstructionDispatchType(type, T) \
		if constexpr(I<B, T, S, false>::TypeSupported) { \
			if (dynamic_cast<const T*>(type)) { \
				visitor.Visit(static_cast<const I<B, T, S>*>(this)); \
			} \
		}

		const auto type = GetType();
		InstructionDispatchType(type);

#undef _InstructionDispatchType
	}

protected:
	virtual const Bits GetBits() const = 0;
	virtual const Type *GetType() const = 0;
	virtual const StateSpace *GetStateSpace() const = 0;
};

class InstructionDispatch_Data2
{
public:
	template<class V, template<Bits, class, class, class, bool = true> class I>
	void Dispatch(V& visitor) const
	{
		const auto bits = GetBits();
		if (bits == Bits::Bits32)
		{
			Dispatch<V, I, Bits::Bits32>(visitor);
		}
		else if (bits == Bits::Bits64)
		{
			Dispatch<V, I, Bits::Bits64>(visitor);
		}
	}

	template<class V, template<Bits, class, class, class, bool = true> class I, Bits B>
	void Dispatch(V& visitor) const
	{
#define _InstructionDispatchSpace(space, S1) \
		if (dynamic_cast<const S1*>(space)) { \
			Dispatch<V, I, B, S1>(visitor); \
		}

		const auto space = GetStateSpace1();
		InstructionDispatchSpace(space);

#undef _InstructionDispatchSpace
	}

	template<class V, template<Bits, class, class, class, bool = true> class I, Bits B, class S1>
	void Dispatch(V& visitor) const
	{
#define _InstructionDispatchSpace(space, S2) \
		if (dynamic_cast<const S2*>(space)) { \
			Dispatch<V, I, B, S1, S2>(visitor); \
		}

		const auto space = GetStateSpace2();
		InstructionDispatchSpace(space);

#undef _InstructionDispatchSpace
	}

	template<class V, template<Bits, class, class, class, bool = true> class I, Bits B, class S1, class S2>
	void Dispatch(V& visitor) const
	{
#define _InstructionDispatchType(type, T) \
		if constexpr(I<B, T, S1, S2, false>::TypeSupported) { \
			if (dynamic_cast<const T*>(type)) { \
				visitor.Visit(static_cast<const I<B, T, S1, S2>*>(this)); \
			} \
		}

		const auto type = GetType();
		InstructionDispatchType(type);

#undef _InstructionDispatchType
	}

protected:
	virtual const Bits GetBits() const = 0;
	virtual const Type *GetType() const = 0;
	virtual const StateSpace *GetStateSpace1() const = 0;
	virtual const StateSpace *GetStateSpace2() const = 0;
};

class InstructionDispatch_DataAtomic
{
public:
	template<class V, template<Bits, class, class, typename, bool = true> class I, typename A>
	void Dispatch(V& visitor) const
	{
		const auto bits = GetBits();
		if (bits == Bits::Bits32)
		{
			Dispatch<V, I, Bits::Bits32, A>(visitor);
		}
		else if (bits == Bits::Bits64)
		{
			Dispatch<V, I, Bits::Bits64, A>(visitor);
		}
	}

	template<class V, template<Bits, class, class, typename, bool = true> class I, Bits B, typename A>
	void Dispatch(V& visitor) const
	{
#define _InstructionDispatchSpace(space, S) \
		if (dynamic_cast<const S*>(space)) { \
			Dispatch<V, I, B, S, A>(visitor); \
		}

		const auto space = GetStateSpace();
		InstructionDispatchSpace(space);

#undef _InstructionDispatchSpace
	}

	template<class V, template<Bits, class, class, typename, bool = true> class I, Bits B, class S, typename A>
	void Dispatch(V& visitor) const
	{
#define _InstructionDispatchType(type, T) \
		if constexpr(I<B, T, S, A, false>::TypeSupported) { \
			if (dynamic_cast<const T*>(type)) { \
				visitor.Visit(static_cast<const I<B, T, S, A>*>(this)); \
			} \
		}

		const auto type = GetType();
		InstructionDispatchType(type);

#undef _InstructionDispatchType
	}

protected:
	virtual const Bits GetBits() const = 0;
	virtual const Type *GetType() const = 0;
	virtual const StateSpace *GetStateSpace() const = 0;
};

#define DispatchInterface(x) \
	class _##x : public InstructionDispatch \
	{ \
	public: \
		template<class V> void Dispatch(V& visitor) const; \
	};
 
#define DispatchInterface_2(x) \
	class _##x : public InstructionDispatch_2 \
	{ \
	public: \
		template<class V> void Dispatch(V& visitor) const; \
	};

#define DispatchInterface_Vector(x) \
	class _##x : public InstructionDispatch_Vector \
	{ \
	public: \
		template<class V> void Dispatch(V& visitor) const; \
	};

#define DispatchInterface_Data(x) \
	class _##x : public InstructionDispatch_Data \
	{ \
	public: \
		template<class V> void Dispatch(V& visitor) const; \
	};

#define DispatchInterface_Data2(x) \
	class _##x : public InstructionDispatch_Data2 \
	{ \
	public: \
		template<class V> void Dispatch(V& visitor) const; \
	};

#define DispatchInterface_DataAtomic(x, y) \
	class _##x : public InstructionDispatch_DataAtomic \
	{ \
	public: \
		template<class V> void Dispatch(V& visitor) const; \
		virtual const y GetAtomic() const = 0; \
	};

#define DispatchImplementation(x) \
	template<class V> \
	void _##x::Dispatch(V& visitor) const \
	{ \
		InstructionDispatch::Dispatch<V, x>(visitor); \
	}

#define DispatchImplementation_2(x) \
	template<class V> \
	void _##x::Dispatch(V& visitor) const \
	{ \
		InstructionDispatch_2::Dispatch<V, x>(visitor); \
	}

#define DispatchImplementation_Vector(x) \
	template<class V> \
	void _##x::Dispatch(V& visitor) const \
	{ \
		InstructionDispatch_Vector::Dispatch<V, x>(visitor); \
	}

#define DispatchImplementation_Data(x) \
	template<class V> \
	void _##x::Dispatch(V& visitor) const \
	{ \
		InstructionDispatch_Data::Dispatch<V, x>(visitor); \
	}

#define DispatchImplementation_Data2(x) \
	template<class V> \
	void _##x::Dispatch(V& visitor) const \
	{ \
		InstructionDispatch_Data2::Dispatch<V, x>(visitor); \
	}

#define DispatchImplementation_DataAtomic(x, y) \
	template<class V> \
	void _##x::Dispatch(V& visitor) const \
	{ \
		y; \
	}

#define DispatchInherit(x) public _##x

#define DispatchMember_Bits(b) \
	const Bits GetBits() const override { return b; }
#define DispatchMember_Type(t) \
	const t *GetType() const override { return &m_type; } \
	t m_type;
#define DispatchMember_Type1(x) \
	const x *GetType1() const override { return &m_type1; } \
	x m_type1;
#define DispatchMember_Type2(x) \
	const x *GetType2() const override { return &m_type2; } \
	x m_type2;
#define DispatchMember_Space(s) \
	const s *GetStateSpace() const override { return &m_space; } \
	s m_space;
#define DispatchMember_Space1(s) \
	const s *GetStateSpace1() const override { return &m_space1; } \
	s m_space1;
#define DispatchMember_Space2(s) \
	const s *GetStateSpace2() const override { return &m_space2; } \
	s m_space2;
#define DispatchMember_Vector(v) \
	const VectorSize GetVectorSize() const override { return v; }
#define DispatchMember_Atomic(t,v) \
	const t GetAtomic() const override { return v; }

}
