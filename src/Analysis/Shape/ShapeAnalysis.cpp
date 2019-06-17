#include "Analysis/Shape/ShapeAnalysis.h"

#include "Analysis/Shape/ShapeAnalysisHelper.h"

#include "Utils/Logger.h"

namespace Analysis {

void ShapeAnalysis::Visit(const HorseIR::AssignStatement *assignS)
{
	// For each target, update the shape with the shape from the expression

	auto expression = assignS->GetExpression();
	auto expressionShapes = ShapeAnalysisHelper::GetShapes(m_currentInSet, expression);

	// Check the number of shapes matches the number of targets

	auto targets = assignS->GetTargets();
	if (expressionShapes.size() != targets.size())
	{
		Utils::Logger::LogError("Mismatched number of shapes for assignment. Received " + std::to_string(expressionShapes.size()) + ", expected " + std::to_string(targets.size()) + ".");
	}

	// Update map for each target symbol

	unsigned int i = 0;
	for (const auto target : targets)
	{
		// Extract the target shape from the expression

		auto symbol = target->GetSymbol();
		m_currentOutSet[symbol] = expressionShapes.at(i++);
	}
}

ShapeAnalysis::Properties ShapeAnalysis::Merge(const Properties& s1, const Properties& s2) const
{
	// Merge the maps using a shape merge operation on each element

	Properties outSet(s1);
	for (const auto val : s2)
	{
		auto it = outSet.find(val.first);
		if (it != outSet.end())
		{
			auto shape1 = val.second;
			auto shape2 = it->second;

			// Merge shapes according to the rules

			outSet[val.first] = MergeShape(shape1, shape2);
		}
		else
		{
			outSet.insert(val);
		}
	}
	return outSet;
}

const Shape *ShapeAnalysis::MergeShape(const Shape* shape1, const Shape *shape2) const
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
			case Shape::Kind::Wildcard:
			{
				return new WildcardShape();
			}
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
				auto mergedShape = MergeShape(listShape1->GetElementShape(), listShape2->GetElementShape());
				return new ListShape(mergedSize, mergedShape);
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

				auto mergedSize = MergeSize(enumShape1->GetMapSize(), enumShape2->GetMapSize());
				return new EnumerationShape(mergedSize);
			}
		}
	}
	
	// If merging fails, return a wildcard

	return new WildcardShape();
}

const Shape::Size *ShapeAnalysis::MergeSize(const Shape::Size *size1, const Shape::Size *size2) const
{
	// Either the sizes are equal, or the size is dynamic

	if (*size1 == *size2)
	{
		return size1;
	}
	return new Shape::DynamicSize();
}

}
