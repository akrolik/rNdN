#include "Runtime/DataBuffers/DataBuffer.h"

#include "HorseIR/Analysis/Shape/ShapeUtils.h"
#include "HorseIR/Utils/TypeUtils.h"

#include "Runtime/DataBuffers/ListCellBuffer.h"
#include "Runtime/DataBuffers/ListCompressedBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"

namespace Runtime {

DataBuffer *DataBuffer::CreateEmpty(const HorseIR::Type *type, const HorseIR::Analysis::Shape *shape)
{
	switch (shape->GetKind())
	{
		case HorseIR::Analysis::Shape::Kind::Vector:
		{
			if (const auto vectorType = HorseIR::TypeUtils::GetType<HorseIR::BasicType>(type))
			{
				auto vectorShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(shape);
				return VectorBuffer::CreateEmpty(vectorType, vectorShape->GetSize());
			}
			break;
		}
		case HorseIR::Analysis::Shape::Kind::List:
		{
			if (const auto listType = HorseIR::TypeUtils::GetType<HorseIR::ListType>(type))
			{
				auto listShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(shape);
				const auto& cellShapes = listShape->GetElementShapes();

				if (cellShapes.size() == 1)
				{
					if (const auto vectorShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(cellShapes.at(0)))
					{
						if (const auto rangedSize = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::RangedSize>(vectorShape->GetSize()))
						{
							auto cellType = HorseIR::TypeUtils::GetReducedType(listType->GetElementTypes());
							if (const auto vectorType = HorseIR::TypeUtils::GetType<HorseIR::BasicType>(cellType))
							{
								return ListCompressedBuffer::CreateEmpty(vectorType, rangedSize);
							}
						}
					}
				}
				return ListCellBuffer::CreateEmpty(listType, listShape);
			}
			break;
		}
	}
	Utils::Logger::LogError("Unable to create empty buffer for shape " + HorseIR::Analysis::ShapeUtils::ShapeString(shape));
}

}
