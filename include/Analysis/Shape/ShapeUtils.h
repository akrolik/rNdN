#pragma once

#include <string>
#include <sstream>
#include <vector>

#include "Analysis/Shape/Shape.h"
#include "HorseIR/Utils/TypeUtils.h"

namespace Analysis {

class ShapeUtils
{
public:

static std::string ShapeString(const Shape *shape)
{
	std::stringstream string;
	string << *shape;
	return string.str();
}	

static std::string SizeString(const Shape::Size *size)
{
	std::stringstream string;
	string << *size;
	return string.str();
}	

static Shape *SymbolicShapeFromType(const HorseIR::Type *type, const std::string& name)
{
	switch (type->m_kind)
	{
		case HorseIR::Type::Kind::Wildcard:
		{
			return new WildcardShape();
		}
		case HorseIR::Type::Kind::Basic:
		{
			return new VectorShape(new Shape::SymbolSize(name));
		}
		case HorseIR::Type::Kind::List:
		{
			auto listType = static_cast<const HorseIR::ListType *>(type);

			std::vector<const Shape *> elementShapes;
			auto index = 0u;
			for (const auto elementType : listType->GetElementTypes())
			{
				auto cellName = name + ".cell" + std::to_string(index++);
				elementShapes.push_back(SymbolicShapeFromType(elementType, cellName));
			}

			if (elementShapes.size() == 1)
			{
				return new ListShape(new Shape::SymbolSize(name + ".cells"), elementShapes);
			}
			return new ListShape(new Shape::ConstantSize(elementShapes.size()), elementShapes);
		}
		case HorseIR::Type::Kind::Table:
		{
			return new TableShape(new Shape::SymbolSize(name + ".cols"), new Shape::SymbolSize(name + ".rows"));
		}
		case HorseIR::Type::Kind::Dictionary:
		{
			auto dictionaryType = static_cast<const HorseIR::DictionaryType *>(type);
			return new DictionaryShape(
				SymbolicShapeFromType(dictionaryType->GetKeyType(), name + ".key"),
				SymbolicShapeFromType(dictionaryType->GetValueType(), name + ".value")
			);
		}
		case HorseIR::Type::Kind::Enumeration:
		{
			auto enumType = static_cast<const HorseIR::EnumerationType *>(type);
			return new EnumerationShape(
				SymbolicShapeFromType(enumType->GetElementType(), name + ".key"),
				SymbolicShapeFromType(enumType->GetElementType(), name + ".value")
			);
		}
		case HorseIR::Type::Kind::KeyedTable:
		{
			auto tableType = static_cast<const HorseIR::KeyedTableType *>(type);

			auto keyTableType = SymbolicShapeFromType(tableType->GetKeyType(), name + ".key");
			auto valueTableType = SymbolicShapeFromType(tableType->GetValueType(), name + ".value");

			return new KeyedTableShape(
				GetShape<TableShape>(keyTableType),
				GetShape<TableShape>(valueTableType)
			);
		}
	}

	Utils::Logger::LogError("Unknown symbolic shape for type " + HorseIR::TypeUtils::TypeString(type));
}

static Shape *InitialShapeFromType(const HorseIR::Type *type)
{
	switch (type->m_kind)
	{
		case HorseIR::Type::Kind::Wildcard:
		{
			return new WildcardShape();
		}
		case HorseIR::Type::Kind::Basic:
		{
			return new VectorShape(new Shape::InitSize());
		}
		case HorseIR::Type::Kind::List:
		{
			auto listType = static_cast<const HorseIR::ListType *>(type);

			std::vector<const Shape *> elementShapes;
			for (const auto elementType : listType->GetElementTypes())
			{
				elementShapes.push_back(InitialShapeFromType(elementType));
			}

			if (elementShapes.size() == 1)
			{
				return new ListShape(new Shape::InitSize(), elementShapes);
			}
			return new ListShape(new Shape::ConstantSize(elementShapes.size()), elementShapes);
		}
		case HorseIR::Type::Kind::Table:
		{
			return new TableShape(new Shape::InitSize(), new Shape::InitSize());
		}
		case HorseIR::Type::Kind::Dictionary:
		{
			auto dictionaryType = static_cast<const HorseIR::DictionaryType *>(type);
			return new DictionaryShape(
				InitialShapeFromType(dictionaryType->GetKeyType()),
				InitialShapeFromType(dictionaryType->GetValueType())
			);
		}
		case HorseIR::Type::Kind::Enumeration:
		{
			auto enumType = static_cast<const HorseIR::EnumerationType *>(type);
			return new EnumerationShape(
				InitialShapeFromType(enumType->GetElementType()),
				InitialShapeFromType(enumType->GetElementType())
			);
		}
		case HorseIR::Type::Kind::KeyedTable:
		{
			auto tableType = static_cast<const HorseIR::KeyedTableType *>(type);

			auto keyTableType = InitialShapeFromType(tableType->GetKeyType());
			auto valueTableType = InitialShapeFromType(tableType->GetValueType());

			return new KeyedTableShape(
				GetShape<TableShape>(keyTableType),
				GetShape<TableShape>(valueTableType)
			);
		}
	}

	Utils::Logger::LogError("Unknown initial shape for type " + HorseIR::TypeUtils::TypeString(type));
}

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

			if (elementShapes.size() == 1)
			{
				return new ListShape(new Shape::DynamicSize(call, tag), elementShapes);
			}
			return new ListShape(new Shape::ConstantSize(elementShapes.size()), elementShapes);
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
	if (!IsShape<VectorShape>(needle) || !IsShape<VectorShape>(haystack))
	{
		return false;
	}

	auto needleSize = GetShape<VectorShape>(needle)->GetSize();
	auto haystackSize = GetShape<VectorShape>(haystack)->GetSize();

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

static bool IsScalar(const Shape *shape)
{
	if (const auto vectorShape = GetShape<VectorShape>(shape))
	{
		return IsScalarSize(vectorShape->GetSize());
	}
	return false;
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
				auto vectorShape1 = GetShape<VectorShape>(shape1);
				auto vectorShape2 = GetShape<VectorShape>(shape2);

				auto mergedSize = MergeSize(vectorShape1->GetSize(), vectorShape2->GetSize());
				return new VectorShape(mergedSize);
			}
			case Shape::Kind::List:
			{
				auto listShape1 = GetShape<ListShape>(shape1);
				auto listShape2 = GetShape<ListShape>(shape2);

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
				auto tableShape1 = GetShape<TableShape>(shape1);
				auto tableShape2 = GetShape<TableShape>(shape2);

				auto mergedColsSize = MergeSize(tableShape1->GetColumnsSize(), tableShape2->GetColumnsSize());
				auto mergedRowsSize = MergeSize(tableShape1->GetRowsSize(), tableShape2->GetRowsSize());
				return new TableShape(mergedColsSize, mergedRowsSize);
			}
			case Shape::Kind::Dictionary:
			{
				auto dictShape1 = GetShape<DictionaryShape>(shape1);
				auto dictShape2 = GetShape<DictionaryShape>(shape2);

				auto mergedKeyShape = MergeShape(dictShape1->GetKeyShape(), dictShape2->GetKeyShape());
				auto mergedValueShape = MergeShape(dictShape1->GetValueShape(), dictShape2->GetValueShape());
				return new DictionaryShape(mergedKeyShape, mergedValueShape);
			}
			case Shape::Kind::Enumeration:
			{
				auto enumShape1 = GetShape<EnumerationShape>(shape1);
				auto enumShape2 = GetShape<EnumerationShape>(shape2);

				auto mergedKeyShape = MergeShape(enumShape1->GetKeyShape(), enumShape2->GetKeyShape());
				auto mergedValueShape = MergeShape(enumShape1->GetValueShape(), enumShape2->GetValueShape());
				return new EnumerationShape(mergedKeyShape, mergedValueShape);
			}
			case Shape::Kind::KeyedTable:
			{
				auto tableShape1 = GetShape<KeyedTableShape>(shape1);
				auto tableShape2 = GetShape<KeyedTableShape>(shape2);

				auto mergedKeyShape = MergeShape(tableShape1->GetKeyShape(), tableShape1->GetKeyShape());
				auto mergedValueShape = MergeShape(tableShape2->GetValueShape(), tableShape2->GetValueShape());

				auto mergedKeyTable = GetShape<TableShape>(mergedKeyShape);
				auto mergedValueTable = GetShape<TableShape>(mergedValueShape);
				return new KeyedTableShape(mergedKeyTable, mergedValueTable);
			}
		}
	}
	
	// If merging fails, return a wildcard

	return new WildcardShape(shape1, shape2);
}

static bool IsDynamicShape(const Shape *shape)
{
	switch (shape->GetKind())
	{
		case Shape::Kind::Vector:
		{
			auto vectorShape = GetShape<VectorShape>(shape);
			return IsDynamicSize(vectorShape->GetSize());
		}
		case Shape::Kind::List:
		{
			auto listShape = GetShape<ListShape>(shape);
			if (IsDynamicSize(listShape->GetListSize()))
			{
				return true;
			}

			for (auto elementShape : listShape->GetElementShapes())
			{
				if (IsDynamicShape(elementShape))
				{
					return true;
				}
			}
			return false;
		}
		case Shape::Kind::Table:
		{
			auto tableShape = GetShape<TableShape>(shape);
			return (IsDynamicSize(tableShape->GetColumnsSize()) || IsDynamicSize(tableShape->GetRowsSize()));
		}
		case Shape::Kind::Dictionary:
		{
			auto dictShape = GetShape<DictionaryShape>(shape);
			return (IsDynamicShape(dictShape->GetKeyShape()) || IsDynamicShape(dictShape->GetValueShape()));
		}
		case Shape::Kind::Enumeration:
		{
			auto enumShape = GetShape<EnumerationShape>(shape);
			return (IsDynamicShape(enumShape->GetKeyShape()) || IsDynamicShape(enumShape->GetValueShape()));
		}
		case Shape::Kind::KeyedTable:
		{
			auto tableShape = GetShape<KeyedTableShape>(shape);
			return (IsDynamicShape(tableShape->GetKeyShape()) || IsDynamicShape(tableShape->GetValueShape()));
		}
	}
	return false;
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
	if (const auto constantSize = GetSize<Shape::ConstantSize>(size))
	{
		return (constantSize->GetValue() == 1);
	}
	return false;
}

static bool IsSubsize(const Shape::Size *needle, const Shape::Size *haystack)
{
	if (const auto needleConstant = GetSize<Shape::ConstantSize>(needle))
	{
		if (const auto haystackConstant = GetSize<Shape::ConstantSize>(haystack))
		{
			return (needleConstant->GetValue() < haystackConstant->GetValue());
		}
		return (needleConstant->GetValue() == 1);
	}

	if (const auto compressedNeedle = GetSize<Shape::CompressedSize>(needle))
	{
		auto unmaskedSize = compressedNeedle->GetSize();
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

static bool IsDynamicSize(const Shape::Size *size)
{
	return (IsSize<Shape::ConstantSize>(size) == false);
}

};

}
