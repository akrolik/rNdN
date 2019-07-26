#pragma once

#include "Analysis/Shape/Shape.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

class Geometry
{
public:
	enum class Kind {
		Unknown,
		CPU,
		Shape
	};

	Geometry(Kind kind) : m_kind(kind) {}

	Kind GetKind() const { return m_kind; }

	bool operator==(const Geometry& other) const;
	bool operator!=(const Geometry& other) const
	{
		return !(*this == other);
	}

private:
	Kind m_kind;
};

class UnknownGeometry : public Geometry
{
public:
	UnknownGeometry(const HorseIR::CallExpression *call) : Geometry(Geometry::Kind::Unknown), m_call(call) {}

	bool operator==(const UnknownGeometry& other) const
	{
		return false;
	}

	bool operator!=(const UnknownGeometry& other) const
	{
		return true;
	}

private:
	const HorseIR::CallExpression *m_call = nullptr;
};

class CPUGeometry : public Geometry
{
public:
	CPUGeometry() : Geometry(Geometry::Kind::CPU) {}

	bool operator==(const CPUGeometry& other) const
	{
		return false;
	}

	bool operator!=(const CPUGeometry& other) const
	{
		return true;
	}
};

class ShapeGeometry : public Geometry
{
public:
	ShapeGeometry(const Shape *shape) : Geometry(Geometry::Kind::Shape), m_shape(shape) {}

	const Shape *GetShape() const { return m_shape; }

	bool operator==(const ShapeGeometry& other) const
	{
		return (*m_shape == *other.m_shape);
	}

	bool operator!=(const ShapeGeometry& other) const
	{
		return !(*this == other);
	}

private:
	const Shape *m_shape = nullptr;
};

inline bool Geometry::operator==(const Geometry& other) const
{
	if (m_kind == other.m_kind)
	{
		switch (m_kind)
		{
			case Geometry::Kind::Unknown:
				return (static_cast<const UnknownGeometry&>(*this) == static_cast<const UnknownGeometry&>(other));
			case Geometry::Kind::CPU:
				return (static_cast<const CPUGeometry&>(*this) == static_cast<const CPUGeometry&>(other));
			case Geometry::Kind::Shape:
				return (static_cast<const ShapeGeometry&>(*this) == static_cast<const ShapeGeometry&>(other));
		}
	}
	return false;
}

}
