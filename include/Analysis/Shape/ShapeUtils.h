#pragma once

#include "Analysis/Shape/Shape.h"

namespace Analysis {

class ShapeUtils
{
public:

template<class S>
static S *GetShape(Shape *shape)
{
	if (shape->GetKind() == S::ShapeKind)
	{
		return static_cast<S *>(shape);
	}
	return nullptr;
}

template<class S>
static const S *GetShape(const Shape *shape)
{
	if (shape->GetKind() == S::ShapeKind)
	{
		return static_cast<const S *>(shape);
	}
	return nullptr;
}

template<class S>
static bool IsShape(const Shape *shape)
{
	return (shape->GetKind() == S::ShapeKind);
}

template<class S>
static S *GetSize(Shape::Size *size)
{
	if (size->GetKind() == S::SizeKind)
	{
		return static_cast<S *>(size);
	}
	return nullptr;
}

template<class S>
static const S *GetSize(const Shape::Size *size)
{
	if (size->GetKind() == S::SizeKind)
	{
		return static_cast<const S *>(size);
	}
	return nullptr;
}

template<class S>
static bool IsSize(const Shape::Size *size)
{
	return (size->GetKind() == S::SizeKind);
}
		
};

}
