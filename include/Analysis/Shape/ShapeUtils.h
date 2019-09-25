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
			auto enumType = static_cast<const HorseIR::EnumerationType *>(type);
			return new EnumerationShape(
				ShapeFromType(enumType->GetElementType(), call, tag),
				ShapeFromType(enumType->GetElementType(), call, tag)
			);
		}
		case HorseIR::Type::Kind::KeyedTable:
		{
			auto tableType = static_cast<const HorseIR::KeyedTableType *>(type);

			auto keyTableType = ShapeFromType(tableType->GetKeyType(), call, tag);
			auto valueTableType = ShapeFromType(tableType->GetValueType(), call, tag);

			return new KeyedTableShape(
				GetShape<TableShape>(keyTableType),
				GetShape<TableShape>(valueTableType)
			);
		}
	}

	Utils::Logger::LogError("Unknown default shape for type " + HorseIR::TypeUtils::TypeString(type));
}
 
static bool IsSubshape(const Shape *needle, const Shape *haystack)
{
	if (!ShapeUtils::IsShape<VectorShape>(needle) || !ShapeUtils::IsShape<VectorShape>(haystack))
	{
		return false;
	}

	auto needleSize = ShapeUtils::GetShape<VectorShape>(needle)->GetSize();
	auto haystackSize = ShapeUtils::GetShape<VectorShape>(haystack)->GetSize();

	return IsSubsize(needleSize, haystackSize);
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

static bool IsSingleShape(const std::vector<const Shape *>& shapes)
{
	return (shapes.size() == 1);
}

static const Shape *GetSingleShape(const std::vector<const Shape *>& shapes)
{
	if (shapes.size() == 1)
	{
		return shapes.at(0);
	}
	return nullptr;
}
static const Shape *MergeShapes(const std::vector<const Shape *>& shapes)
{
	const Shape *mergedShape = nullptr;
	for (auto& shape : shapes)
	{
		if (mergedShape == nullptr)
		{
			mergedShape = shape;
		}
		else if (*mergedShape != *shape)
		{
			mergedShape = MergeShape(mergedShape, shape);
		}
	}
	return mergedShape;
}

static const Shape::Size *GetMappingSize(const Shape *shape)
{
	switch (shape->GetKind())
	{
		case Shape::Kind::Vector:
		{
			return GetShape<VectorShape>(shape)->GetSize();
		}
		case Shape::Kind::List:
		{
			return GetShape<ListShape>(shape)->GetListSize();
		}
		case Shape::Kind::Dictionary:
		case Shape::Kind::Enumeration:
		case Shape::Kind::Table:
		case Shape::Kind::KeyedTable:
		{
			return new Shape::ConstantSize(1);
		}
	}
	return new Shape::ConstantSize(1);
}

static const Shape *MergeShape(const Shape *shape1, const Shape *shape2)
{
	if (*shape1 == *shape2)
	{
		return shape1;
	}

	if (shape1->GetKind() == shape2->GetKind())
	{
		// If the shapes are equal kind, merge the contents recursively

		switch (shape1->GetKind())
		{
			case Shape::Kind::Vector:
			{
				auto vectorShape1 = static_cast<const VectorShape *>(shape1);
				auto vectorShape2 = static_cast<const VectorShape *>(shape2);

				auto mergedSize = MergeSize(vectorShape1->GetSize(), vectorShape2->GetSize());
				return new VectorShape(mergedSize);
			}
			case Shape::Kind::List:
			{
				auto listShape1 = static_cast<const ListShape *>(shape1);
				auto listShape2 = static_cast<const ListShape *>(shape2);

				auto mergedSize = MergeSize(listShape1->GetListSize(), listShape2->GetListSize());

				auto elementShapes1 = listShape1->GetElementShapes();
				auto elementShapes2 = listShape2->GetElementShapes();

				std::vector<const Shape *> mergedElementShapes;
				if (elementShapes1.size() == elementShapes2.size())
				{
					unsigned int i = 0;
					for (const auto elementShape1 : elementShapes1)
					{
						const auto elementShape2 = elementShapes2.at(i++);
						auto mergedShape = MergeShape(elementShape1, elementShape2);
						mergedElementShapes.push_back(mergedShape);
					}
				}
				else
				{
					mergedElementShapes.push_back(new WildcardShape(shape1, shape2));
				}
				return new ListShape(mergedSize, mergedElementShapes);
			}
			case Shape::Kind::Table:
			{
				auto tableShape1 = static_cast<const TableShape *>(shape1);
				auto tableShape2 = static_cast<const TableShape *>(shape2);

				auto mergedColsSize = MergeSize(tableShape1->GetColumnsSize(), tableShape2->GetColumnsSize());
				auto mergedRowsSize = MergeSize(tableShape1->GetRowsSize(), tableShape2->GetRowsSize());
				return new TableShape(mergedColsSize, mergedRowsSize);
			}
			case Shape::Kind::Dictionary:
			{
				auto dictShape1 = static_cast<const DictionaryShape *>(shape1);
				auto dictShape2 = static_cast<const DictionaryShape *>(shape2);

				auto mergedKeyShape = MergeShape(dictShape1->GetKeyShape(), dictShape2->GetKeyShape());
				auto mergedValueShape = MergeShape(dictShape1->GetValueShape(), dictShape2->GetValueShape());
				return new DictionaryShape(mergedKeyShape, mergedValueShape);
			}
			case Shape::Kind::Enumeration:
			{
				auto enumShape1 = static_cast<const EnumerationShape *>(shape1);
				auto enumShape2 = static_cast<const EnumerationShape *>(shape2);

				auto mergedKeyShape = MergeShape(enumShape1->GetKeyShape(), enumShape2->GetKeyShape());
				auto mergedValueShape = MergeShape(enumShape1->GetValueShape(), enumShape2->GetValueShape());
				return new EnumerationShape(mergedKeyShape, mergedValueShape);
			}
			case Shape::Kind::KeyedTable:
			{
				auto tableShape1 = static_cast<const KeyedTableShape *>(shape1);
				auto tableShape2 = static_cast<const KeyedTableShape *>(shape2);

				auto mergedKeyShape = MergeShape(tableShape1->GetKeyShape(), tableShape1->GetKeyShape());
				auto mergedValueShape = MergeShape(tableShape2->GetValueShape(), tableShape2->GetValueShape());

				auto mergedKeyTable = static_cast<const TableShape *>(mergedKeyShape);
				auto mergedValueTable = static_cast<const TableShape *>(mergedValueShape);
				return new KeyedTableShape(mergedKeyTable, mergedValueTable);
			}
		}
	}
	
	// If merging fails, return a wildcard

	return new WildcardShape(shape1, shape2);
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
		
static bool IsScalarSize(const Shape::Size *size)
{
	if (ShapeUtils::IsSize<Shape::ConstantSize>(size))
	{
		return (ShapeUtils::GetSize<Shape::ConstantSize>(size)->GetValue() == 1);
	}
	return false;
}

static bool IsSubsize(const Shape::Size *needle, const Shape::Size *haystack)
{
	if (ShapeUtils::IsSize<Shape::ConstantSize>(needle))
	{
		return (ShapeUtils::GetSize<Shape::ConstantSize>(needle)->GetValue() == 1);
	}

	if (ShapeUtils::IsSize<Shape::CompressedSize>(needle))
	{
		auto unmaskedSize = ShapeUtils::GetSize<Shape::CompressedSize>(needle)->GetSize();
		if (*unmaskedSize == *haystack)
		{
			return true;
		}
		return IsSubsize(unmaskedSize, haystack);
	}

	return false;
}

static const Shape::Size *MergeSize(const Shape::Size *size1, const Shape::Size *size2)
{
	// Either the sizes are equal, or the size is dynamic

	if (*size1 == *size2)
	{
		return size1;
	}
	return new Shape::DynamicSize(size1, size2);
}

};

}
