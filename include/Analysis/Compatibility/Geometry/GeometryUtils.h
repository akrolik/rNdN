#pragma once

#include "Analysis/Compatibility/Geometry/Geometry.h"
#include "Analysis/Shape/ShapeUtils.h"

namespace Analysis {

class GeometryUtils
{
public:

static bool IsSubgeometry(const Geometry *needle, const Geometry *haystack)
{
	if (needle->GetKind() != Geometry::Kind::Shape || haystack->GetKind() != Geometry::Kind::Shape)
	{
		return false;
	}

	auto needleShape = static_cast<const ShapeGeometry *>(needle)->GetShape();
	auto haystackShape = static_cast<const ShapeGeometry *>(haystack)->GetShape();

	return ShapeUtils::IsSubshape(needleShape, haystackShape);
}

static const Geometry *MaxGeometry(const Geometry *geometry1, const Geometry *geometry2)
{
	if (geometry1 == nullptr)
	{
		return geometry2;
	}
	else if (geometry2 == nullptr)
	{
		return geometry1;
	}

	if (IsSubgeometry(geometry1, geometry2))
	{
		return geometry2;
	}
	return geometry1;
}

};

}
