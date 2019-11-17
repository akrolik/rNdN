#pragma once

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

namespace Runtime {

class RuntimeUtils
{
public:
	static bool IsDynamicReturnShape(const Analysis::Shape *dataShape, const Analysis::Shape *threadGeometry)
	{
		// Statically defined output shapes

		if (!Analysis::ShapeUtils::IsDynamicShape(dataShape))
		{
			return false;
		}

		// Geometry defined output shape

		if (*dataShape == *threadGeometry)
		{
			return false;
		}

		// @raze special case output shape

		if (const auto listGeometry = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(threadGeometry))
		{
			// Ensure vector cell geometry

			const auto cellGeometry = Analysis::ShapeUtils::MergeShapes(listGeometry->GetElementShapes());
			if (const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellGeometry))
			{
				if (const auto vectorData = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(dataShape))
				{
					// Special case of @raze returning a vector

					if (*listGeometry->GetListSize() == *vectorData->GetSize())
					{
						return false;
					}
				}
			}
		}

		return true;
	}
};

}
