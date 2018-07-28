#pragma once

#include <map>
#include <unordered_map>

#include "HorseIR/Traversal/ForwardTraversal.h"

#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

struct Shape
{
	constexpr static long DynamicSize = -1;

	enum class Kind {
		Vector,
		List,
		Table
	};

	Shape(Kind k, long s) : kind(k), size(s) {}

	std::string ToString() const
	{
		std::string output = "Shape(";
		switch (kind)
		{
			case Kind::Vector:
				output += "vector";
				break;
			case Kind::List:
				output += "list";
				break;
			case Kind::Table:
				output += "table";
				break;
		}
		output += ", ";
		if (size == DynamicSize)
		{
			output += "DynamicSize";
		}
		else
		{
			output += std::to_string(size);
		}
		return output + ")";
	}

	Kind kind;
	long size;
};

using ShapeMap = std::map<std::string, const Shape *>;
using ExpressionMap = std::unordered_map<const Expression *, const Shape *>;

class ShapeAnalysis : public ForwardTraversal
{
public:
	void SetInputShape(const Parameter *parameter, Shape *shape);

	void Analyze(Method *method);
	const Shape *GetShape(const std::string& identifier) const;

	void Visit(Parameter *parameter) override;
	void Visit(AssignStatement *assign) override;

	void Visit(CallExpression *call);
	void Visit(CastExpression *cast);
	void Visit(Identifier *identifier);
	void Visit(Literal<int64_t> *literal);
	void Visit(Literal<double> *literal);
	void Visit(Literal<std::string> *literal);
	void Visit(Symbol *symbol);

	void Dump() const;

private:
	const Shape *GetShape(const Expression *expression) const;

	void SetShape(const Expression *expression, const Shape *shape);
	void SetShape(const std::string &identifier, const Shape *shape);

	ShapeMap m_identifierMap;
	ExpressionMap m_expressionMap;
};

}
