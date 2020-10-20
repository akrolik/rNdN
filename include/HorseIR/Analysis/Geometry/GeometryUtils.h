#pragma once

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

namespace HorseIR {
namespace Analysis {

class GeometryUtils
{
public:

static const Shape *MaxGeometry(const Shape *geometry1, const Shape *geometry2)
{
	if (geometry1 == nullptr)
	{
		return geometry2;
	}
	else if (geometry2 == nullptr)
	{
		return geometry1;
	}

	if (*geometry1 == *geometry2)
	{
		return geometry1;
	}

	if (geometry1->GetKind() == geometry2->GetKind())
	{
		// If the shapes are equal kind, maximize the contents recursively

		switch (geometry1->GetKind())
		{
			case Shape::Kind::Vector:
			{
				auto vectorShape1 = ShapeUtils::GetShape<VectorShape>(geometry1);
				auto vectorShape2 = ShapeUtils::GetShape<VectorShape>(geometry2);

				auto maxSize = MaxSize(vectorShape1->GetSize(), vectorShape2->GetSize());
				return new VectorShape(maxSize);
			}
			case Shape::Kind::List:
			{
				auto listShape1 = ShapeUtils::GetShape<ListShape>(geometry1);
				auto listShape2 = ShapeUtils::GetShape<ListShape>(geometry2);

				auto maxSize = MaxSize(listShape1->GetListSize(), listShape2->GetListSize());

				auto elementShapes1 = listShape1->GetElementShapes();
				auto elementShapes2 = listShape2->GetElementShapes();

				auto elementCount1 = elementShapes1.size();
				auto elementCount2 = elementShapes2.size();

				std::vector<const Shape *> maxElementShapes;
				if (elementCount1 == elementCount2 || elementCount1 == 1 || elementCount2 == 1)
				{
					auto count = std::max({elementCount1, elementCount2});
					for (auto i = 0u; i < count; ++i)
					{
						const auto l_elementShape1 = elementShapes1.at((elementCount1 == 1) ? 0 : i);
						const auto l_elementShape2 = elementShapes2.at((elementCount2 == 1) ? 0 : i);

						const auto maxShape = MaxGeometry(l_elementShape1, l_elementShape2);
						maxElementShapes.push_back(maxShape);
					}
				}
				else
				{
					maxElementShapes.push_back(new WildcardShape(geometry1, geometry2));
				}
				return new ListShape(maxSize, maxElementShapes);
			}
			case Shape::Kind::Table:
			{
				auto tableShape1 = ShapeUtils::GetShape<TableShape>(geometry1);
				auto tableShape2 = ShapeUtils::GetShape<TableShape>(geometry2);

				auto maxColsSize = MaxSize(tableShape1->GetColumnsSize(), tableShape2->GetColumnsSize());
				auto maxRowsSize = MaxSize(tableShape1->GetRowsSize(), tableShape2->GetRowsSize());
				return new TableShape(maxColsSize, maxRowsSize);
			}
			case Shape::Kind::Dictionary:
			{
				auto dictShape1 = ShapeUtils::GetShape<DictionaryShape>(geometry1);
				auto dictShape2 = ShapeUtils::GetShape<DictionaryShape>(geometry2);

				auto maxKeyShape = MaxGeometry(dictShape1->GetKeyShape(), dictShape2->GetKeyShape());
				auto maxValueShape = MaxGeometry(dictShape1->GetValueShape(), dictShape2->GetValueShape());
				return new DictionaryShape(maxKeyShape, maxValueShape);
			}
			case Shape::Kind::Enumeration:
			{
				auto enumShape1 = ShapeUtils::GetShape<EnumerationShape>(geometry1);
				auto enumShape2 = ShapeUtils::GetShape<EnumerationShape>(geometry2);

				auto maxKeyShape = MaxGeometry(enumShape1->GetKeyShape(), enumShape2->GetKeyShape());
				auto maxValueShape = MaxGeometry(enumShape1->GetValueShape(), enumShape2->GetValueShape());
				return new EnumerationShape(maxKeyShape, maxValueShape);
			}
			case Shape::Kind::KeyedTable:
			{
				auto tableShape1 = ShapeUtils::GetShape<KeyedTableShape>(geometry1);
				auto tableShape2 = ShapeUtils::GetShape<KeyedTableShape>(geometry2);

				auto maxKeyShape = MaxGeometry(tableShape1->GetKeyShape(), tableShape1->GetKeyShape());
				auto maxValueShape = MaxGeometry(tableShape2->GetValueShape(), tableShape2->GetValueShape());

				auto maxKeyTable = ShapeUtils::GetShape<TableShape>(maxKeyShape);
				auto maxValueTable = ShapeUtils::GetShape<TableShape>(maxValueShape);
				return new KeyedTableShape(maxKeyTable, maxValueTable);
			}
		}
	}
	
	// If merging fails, return a wildcard

	return new WildcardShape(geometry1, geometry2);
}

static const Shape::Size *MaxSize(const Shape::Size *size1, const Shape::Size *size2)
{
	// Check for initialization

	if (ShapeUtils::IsSize<Shape::InitSize>(size1))
	{
		return size2;
	}

	if (ShapeUtils::IsSize<Shape::InitSize>(size2))
	{
		return size1;
	}

	// Check for subsizes

	if (ShapeUtils::IsSubsize(size1, size2))
	{
		return size2;
	}
	
	if (ShapeUtils::IsSubsize(size2, size1))
	{
		return size1;
	}

	// Either the sizes are equal, or the size is dynamic

	if (*size1 == *size2)
	{
		return size1;
	}
	return new Shape::DynamicSize(size1, size2);
}

};

}
}
