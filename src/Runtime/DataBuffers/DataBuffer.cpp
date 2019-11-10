#include "Runtime/DataBuffers/DataBuffer.h"

#include "Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Utils/TypeUtils.h"

#include "Runtime/DataBuffers/ListBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

namespace Runtime {

DataBuffer *DataBuffer::CreateEmpty(const HorseIR::Type *type, const Analysis::Shape *shape)
{
	switch (shape->GetKind())
	{
		case Analysis::Shape::Kind::Vector:
		{
			if (const auto vectorType = HorseIR::TypeUtils::GetType<HorseIR::BasicType>(type))
			{
				auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(shape);
				return VectorBuffer::CreateEmpty(vectorType, vectorShape->GetSize());
			}
			Utils::Logger::LogError("Vector shape requires basic type");
		}
		case Analysis::Shape::Kind::List:
		{
			if (const auto listType = HorseIR::TypeUtils::GetType<HorseIR::ListType>(type))
			{
				auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape);
				return ListBuffer::CreateEmpty(listType, listShape);
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
