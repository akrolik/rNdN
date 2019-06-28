#pragma once

#include <vector>

#include "Analysis/Shape/Shape.h"
#include "HorseIR/Utils/TypeUtils.h"

namespace Analysis {

class ShapeUtils
{
public:

static Shape *ShapeFromType(const HorseIR::Type *type, const HorseIR::CallExpression *call = nullptr, unsigned int tag = 0)
{
	switch (type->m_kind)
	{
		case HorseIR::Type::Kind::Wildcard:
		{
			return new WildcardShape();
		}
		case HorseIR::Type::Kind::Basic:
		{
			return new VectorShape(new Shape::DynamicSize(call, tag));
		}
		case HorseIR::Type::Kind::List:
		{
			auto listType = static_cast<const HorseIR::ListType *>(type);

			std::vector<const Shape *> elementShapes;
			for (const auto elementType : listType->GetElementTypes())
			{
				elementShapes.push_back(ShapeFromType(elementType, call, tag));
			}

			return new ListShape(new Shape::DynamicSize(call, tag), elementShapes);
		}
		case HorseIR::Type::Kind::Table:
		{
			return new TableShape(new Shape::DynamicSize(call, tag), new Shape::DynamicSize(call, tag));
		}
		case HorseIR::Type::Kind::Dictionary:
		{
			auto dictionaryType = static_cast<const HorseIR::DictionaryType *>(type);
			return new DictionaryShape(
				ShapeFromType(dictionaryType->GetKeyType(), call, tag),
				ShapeFromType(dictionaryType->GetValueType(), call, tag)
			);
		}
		case HorseIR::Type::Kind::Enumeration:
		{
			return new EnumerationShape(new Shape::DynamicSize(call, tag));
		}
		case HorseIR::Type::Kind::KeyedTable:
		{
			return new KeyedTableShape(
				new TableShape(new Shape::DynamicSize(call, tag), new Shape::DynamicSize(call, tag)),
				new TableShape(new Shape::DynamicSize(call, tag), new Shape::DynamicSize(call, tag))
			);
		}
	}

	Utils::Logger::LogError("Unknown default shape for type " + HorseIR::TypeUtils::TypeString(type));
}

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
