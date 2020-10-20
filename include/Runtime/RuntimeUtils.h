#pragma once

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

namespace Runtime {

class RuntimeUtils
{
public:
	static bool IsDynamicReturnShape(const HorseIR::Analysis::Shape *dataShape, const HorseIR::Analysis::Shape *writeShape, const HorseIR::Analysis::Shape *threadGeometry)
	{
		// Statically defined output shapes

		if (!HorseIR::Analysis::ShapeUtils::IsDynamicShape(dataShape))
		{
			return false;
		}

		// Geometry defined output shape

		if (*dataShape == *threadGeometry)
		{
			return false;
		}

		// @index_a special case 

		if (*dataShape != *writeShape)
		{
			return false;
		}

		// List in vector

		if (const auto vectorGeometry = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(threadGeometry))
		{
			if (const auto listData = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(dataShape))
			{
				const auto cellData = HorseIR::Analysis::ShapeUtils::MergeShapes(listData->GetElementShapes());
				if (const auto vectorData = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(cellData))
				{
					if (*vectorGeometry == *vectorData)
					{
						return false;
					}
				}
			}
		}

		// @raze special case output shape

		if (const auto listGeometry = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(threadGeometry))
		{
			// Ensure vector cell geometry

			const auto cellGeometry = HorseIR::Analysis::ShapeUtils::MergeShapes(listGeometry->GetElementShapes());
			if (const auto vectorGeometry = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(cellGeometry))
			{
				if (const auto vectorData = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(dataShape))
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
