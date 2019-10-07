#pragma once

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

namespace Analysis {

class GeometryUtils
{
public:

static const Shape *MaxGeometry(const Shape *geometry1, const Shape *geometry2)
{
	if (geometry1 == nullptr)
	{
		return geometry2;
	}
	else if (geometry2 == nullptr)
	{
		return geometry1;
	}

	if (ShapeUtils::IsSubshape(geometry1, geometry2))
	{
		return geometry2;
	}
	return geometry1;
}

};

}
