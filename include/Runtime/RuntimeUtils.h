#pragma once

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

namespace Runtime {

class RuntimeUtils
{
public:
	static bool IsDynamicDataShape(const Analysis::Shape *dataShape, const Analysis::Shape *threadGeometry)
	{
		if (const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(threadGeometry))
		{
			// Static vectors (constant) are always statically determined

			if (!Analysis::ShapeUtils::IsDynamicShape(dataShape))
			{
				return false;
			}

			// If the vector geometry is the same as the data shape, we can load vector-wise

			if (*dataShape == *threadGeometry)
			{
				return false;
			}

			// Check for compression loading

			if (const auto vectorShape = Analysis::ShapeUtils::GetShape<VectorShape>(dataShape))
			{
				return !Analysis::ShapeUtils::IsCompressedSize(vectorShape->GetSize(), vectorGeometry->GetSize());
			}
		}
		else if (const auto listGeometry = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(threadGeometry))
		{
			// Ensure vector cell geometry

			const auto cellGeometry = Analysis::ShapeUtils::MergeShapes(listGeometry->GetElementShapes());
			if (const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellGeometry))
			{
				if (const auto vectorData = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(dataShape))
				{
					// If a vector is loaded in list geometry, we can eliminate the size if the cell
					// geometry matches that of the vector

					if (*vectorData == *vectorGeometry)
					{
						return false;
					}
				}
				else if (const auto listData = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(dataShape))
				{
					// List data is constant size if all cells have the same size, and the data vector shape
					// is constant wrt the geometry vector

					const auto cellData = Analysis::ShapeUtils::MergeShapes(listData->GetElementShapes());
					if (const auto vectorData = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellData))
					{
						return IsDynamicDataShape(vectorData, vectorGeometry);
					}
				}
			}
		}

		return true;
	}
};

}
