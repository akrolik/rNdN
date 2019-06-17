#pragma once

#include "Analysis/Shape/Shape.h"

namespace Analysis {

template<class S>
static S *CastShape(Shape *shape)
{
	if (shape->GetKind() == S::ShapeKind)
	{
		return static_cast<S *>(shape);
	}
	return nullptr;
}

template<class S>
static const S *CastShape(const Shape *shape)
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

}
