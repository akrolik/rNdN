#include "Runtime/DataBuffers/DataBuffer.h"

#include "Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Utils/TypeUtils.h"

#include "Runtime/DataBuffers/VectorBuffer.h"

namespace Runtime {

DataBuffer *DataBuffer::Create(const HorseIR::Type *type, const Analysis::Shape *shape)
{
	switch (shape->GetKind())
	{
		case Analysis::Shape::Kind::Vector:
		{
			if (const auto vectorType = HorseIR::TypeUtils::GetType<HorseIR::BasicType>(type))
			{
				auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(shape);
				return VectorBuffer::Create(vectorType, vectorShape);
			}
			Utils::Logger::LogError("Vector shape requires basic type");
		}
		default:
		{
			Utils::Logger::LogError("Unsupported buffer shape and type");
		}
	}
}

}
