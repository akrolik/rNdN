#pragma once

#include "HorseIR/Tree/MethodDeclaration.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"

#include "Utils/Logger.h"

namespace HorseIR {

class BuiltinMethod : public MethodDeclaration
{
public:
	enum class Kind {
		// Unary
		Absolute,
		Negate,
		Ceiling,
		Floor,
		Round,
		Conjugate,
		Reciprocal,
		Sign,
		Pi,
		Not,
		Logarithm,
		Logarithm2,
		Logarithm10,
		SquareRoot,
		Exponential,
		Date,
		DateYear,
		DateMonth,
		DateDay,
		Time,
		TimeHour,
		TimeMinute,
		TimeSecond,
		TimeMillisecond,
		Cosine,
		Sine,
		Tangent,
		InverseCosine,
		InverseSine,
		InverseTangent,
		HyperbolicCosine,
		HyperbolicSine,
		HyperbolicTangent,
		HyperbolicInverseCosine,
		HyperbolicInverseSine,
		HyperbolicInverseTangent,

		// Binary
		Less,
		Greater,
		LessEqual,
		GreaterEqual,
		Equal,
		NotEqual,
		Plus,
		Minus,
		Multiply,
		Divide,
		Power,
		LogarithmBase,
		Modulo,
		And,
		Or,
		Nand,
		Nor,
		Xor,
		DatetimeDifference,

		// Algebraic Unary
		Unique,
		String,
		Length,
		Range,
		Factorial,
		Random,
		Seed,
		Flip,
		Reverse,
		Where,
		Group,

		// Algebraic Binary
		Append,
		Like,
		Compress,
		Random_k,
		IndexOf,
		Take,
		Drop,
		Order,
		Member,
		Vector,

		// Reduction
		Count,
		Sum,
		Average,
		Minimum,
		Maximum,

		// List
		Raze,
		List,
		Enlist,
		ToList,
		Each,
		EachItem,
		EachLeft,
		EachRight,
		Match,
		Outer,

		// Database
		Enum,
		Dictionary,
		Table,
		KeyedTable,
		Keys,
		Values,
		Meta,
		Fetch,
		ColumnValue,
		LoadTable,
		DatetimeAdd,
		DatetimeSubtract,
		JoinIndex,

		// Indexing
		Index,
		IndexAssignment,

		// Other
		LoadCSV,
		Print,
		Format,
		SubString
	};

	static std::string BuiltinMethodName(Kind kind)
	{
		switch (kind)
		{
			case Kind::Absolute:
				return "abs";
			case Kind::Negate:
				return "neg";
			case Kind::Ceiling:
				return "ceil";
			case Kind::Floor:
				return "floor";
			case Kind::Round:
				return "round";
			case Kind::Conjugate:
				return "conj";
			case Kind::Reciprocal:
				return "recip";
			case Kind::Sign:
				return "signum";
			case Kind::Pi:
				return "pi";
			case Kind::Not:
				return "not";
			case Kind::Logarithm:
				return "log";
			case Kind::Logarithm2:
				return "log2";
			case Kind::Logarithm10:
				return "log10";
			case Kind::SquareRoot:
				return "sqrt";
			case Kind::Exponential:
				return "exp";
			case Kind::Date:
				return "date";
			case Kind::DateYear:
				return "date_year";
			case Kind::DateMonth:
				return "date_month";
			case Kind::DateDay:
				return "date_day";
			case Kind::Time:
				return "time";
			case Kind::TimeHour:
				return "time_hour";
			case Kind::TimeMinute:
				return "time_minute";
			case Kind::TimeSecond:
				return "time_second";
			case Kind::TimeMillisecond:
				return "time_mill";
			case Kind::Cosine:
				return "cos";
			case Kind::Sine:
				return "sin";
			case Kind::Tangent:
				return "tan";
			case Kind::InverseCosine:
				return "acos";
			case Kind::InverseSine:
				return "asin";
			case Kind::InverseTangent:
				return "atan";
			case Kind::HyperbolicCosine:
				return "cosh";
			case Kind::HyperbolicSine:
				return "sinh";
			case Kind::HyperbolicTangent:
				return "tanh";
			case Kind::HyperbolicInverseCosine:
				return "acosh";
			case Kind::HyperbolicInverseSine:
				return "asinh";
			case Kind::HyperbolicInverseTangent:
				return "atanh";
			case Kind::Less:
				return "lt";
			case Kind::Greater:
				return "gt";
			case Kind::LessEqual:
				return "leq";
			case Kind::GreaterEqual:
				return "geq";
			case Kind::Equal:
				return "eq";
			case Kind::NotEqual:
				return "neq";
			case Kind::Plus:
				return "plus";
			case Kind::Minus:
				return "minus";
			case Kind::Multiply:
				return "mul";
			case Kind::Divide:
				return "div";
			case Kind::Power:
				return "power";
			case Kind::LogarithmBase:
				return "logb";
			case Kind::Modulo:
				return "mod";
			case Kind::And:
				return "and";
			case Kind::Or:
				return "or";
			case Kind::Nand:
				return "nand";
			case Kind::Nor:
				return "nor";
			case Kind::Xor:
				return "xor";
			case Kind::DatetimeDifference:
				return "datetime_diff";
			case Kind::Unique:
				return "unique";
			case Kind::String:
				return "str";
			case Kind::Length:
				return "len";
			case Kind::Range:
				return "range";
			case Kind::Factorial:
				return "fact";
			case Kind::Random:
				return "rand";
			case Kind::Seed:
				return "seed";
			case Kind::Flip:
				return "flip";
			case Kind::Reverse:
				return "reverse";
			case Kind::Where:
				return "where";
			case Kind::Group:
				return "group";
			case Kind::Append:
				return "append";
			case Kind::Like:
				return "like";
			case Kind::Compress:
				return "compress";
			case Kind::Random_k:
				return "randk";
			case Kind::IndexOf:
				return "index_of";
			case Kind::Take:
				return "take";
			case Kind::Drop:
				return "drop";
			case Kind::Order:
				return "order";
			case Kind::Member:
				return "member";
			case Kind::Vector:
				return "vector";
			case Kind::Count:
				return "count";
			case Kind::Sum:
				return "sum";
			case Kind::Average:
				return "avg";
			case Kind::Minimum:
				return "min";
			case Kind::Maximum:
				return "max";
			case Kind::Raze:
				return "raze";
			case Kind::List:
				return "list";
			case Kind::Enlist:
				return "enlist";
			case Kind::ToList:
				return "tolist";
			case Kind::Each:
				return "each";
			case Kind::EachItem:
				return "each_item";
			case Kind::EachLeft:
				return "each_left";
			case Kind::EachRight:
				return "each_right";
			case Kind::Match:
				return "match";
			case Kind::Outer:
				return "outer";
			case Kind::Enum:
				return "enum";
			case Kind::Dictionary:
				return "dict";
			case Kind::Table:
				return "table";
			case Kind::KeyedTable:
				return "ktable";
			case Kind::Keys:
				return "keys";
			case Kind::Values:
				return "values";
			case Kind::Meta:
				return "meta";
			case Kind::Fetch:
				return "fetch";
			case Kind::ColumnValue:
				return "column_value";
			case Kind::LoadTable:
				return "load_table";
			case Kind::DatetimeAdd:
				return "datetime_add";
			case Kind::DatetimeSubtract:
				return "datetime_sub";
			case Kind::JoinIndex:
				return "join_index";
			case Kind::Index:
				return "index";
			case Kind::IndexAssignment:
				return "index_a";
			case Kind::LoadCSV:
				return "load_csv";
			case Kind::Print:
				return "print";
			case Kind::Format:
				return "format";
			case Kind::SubString:
				return "sub_string";
		}
		return "<unknown>";
	}

	BuiltinMethod(Kind kind) : MethodDeclaration(MethodDeclaration::Kind::Builtin, BuiltinMethodName(kind)), m_kind(kind) {}

	Kind GetKind() const { return m_kind; }

	std::string SignatureString() const override
	{
		return "def " + m_name + "() __BUILTIN__";
	}

	constexpr static int VariadicParameterCount = -1;

	int GetParameterCount() const
	{
		switch (m_kind)
		{
			// Unary functions
			case Kind::Absolute:
			case Kind::Negate:
			case Kind::Ceiling:
			case Kind::Floor:
			case Kind::Round:
			case Kind::Conjugate:
			case Kind::Reciprocal:
			case Kind::Sign:
			case Kind::Pi:
			case Kind::Not:
			case Kind::Logarithm:
			case Kind::Logarithm2:
			case Kind::Logarithm10:
			case Kind::SquareRoot:
			case Kind::Exponential:
			case Kind::Date:
			case Kind::DateYear:
			case Kind::DateMonth:
			case Kind::DateDay:
			case Kind::Time:
			case Kind::TimeHour:
			case Kind::TimeMinute:
			case Kind::TimeSecond:
			case Kind::TimeMillisecond:
			case Kind::Cosine:
			case Kind::Sine:
			case Kind::Tangent:
			case Kind::InverseCosine:
			case Kind::InverseSine:
			case Kind::InverseTangent:
			case Kind::HyperbolicCosine:
			case Kind::HyperbolicSine:
			case Kind::HyperbolicTangent:
			case Kind::HyperbolicInverseCosine:
			case Kind::HyperbolicInverseSine:
			case Kind::HyperbolicInverseTangent:
				return 1;

			// Binary functions
			case Kind::Less:
			case Kind::Greater:
			case Kind::LessEqual:
			case Kind::GreaterEqual:
			case Kind::Equal:
			case Kind::NotEqual:
			case Kind::Plus:
			case Kind::Minus:
			case Kind::Multiply:
			case Kind::Divide:
			case Kind::Power:
			case Kind::LogarithmBase:
			case Kind::Modulo:
			case Kind::And:
			case Kind::Or:
			case Kind::Nand:
			case Kind::Nor:
			case Kind::Xor:
			case Kind::DatetimeDifference:
				return 2;
			
			// Algebraic unary functions
			case Kind::Unique:
			case Kind::String:
			case Kind::Length:
			case Kind::Range:
			case Kind::Factorial:
			case Kind::Random:
			case Kind::Seed:
			case Kind::Flip:
			case Kind::Reverse:
			case Kind::Where:
			case Kind::Group:
				return 1;

			// Algebraic binary functions
			case Kind::Append:
			case Kind::Like:
			case Kind::Compress:
			case Kind::Random_k:
			case Kind::IndexOf:
			case Kind::Take:
			case Kind::Drop:
			case Kind::Order:
			case Kind::Member:
			case Kind::Vector:
				return 2;

			// Reduction functions
			case Kind::Count:
			case Kind::Sum:
			case Kind::Average:
			case Kind::Minimum:
			case Kind::Maximum:
				return 1;

			// List functions
			case Kind::Raze:
				return 1;
			case Kind::List:
				return VariadicParameterCount;
			case Kind::Enlist:
			case Kind::ToList:
				return 1;
			case Kind::Each:
				return 2;
			case Kind::EachItem:
			case Kind::EachLeft:
			case Kind::EachRight:
				return 3;
			case Kind::Match:
				return 2;
			case Kind::Outer:
				return 3;

			// Database functions
			case Kind::Enum:
			case Kind::Dictionary:
			case Kind::Table:
			case Kind::KeyedTable:
				return 2;
			case Kind::Keys:
				return 1;
			case Kind::Values:
			case Kind::Meta:
				return 1;
			case Kind::ColumnValue:
				return 2;
			case Kind::LoadTable:
				return 1;
			case Kind::Fetch:
				return 1;
			case Kind::DatetimeAdd:
			case Kind::DatetimeSubtract:
				return 3;
			case Kind::JoinIndex:
				return 2;

			// Indexing functions
			case Kind::Index:
				return 2;
			case Kind::IndexAssignment:
				return 3;

			// Others
			case Kind::LoadCSV:
			case Kind::Print:
			case Kind::Format:
				return 1;
			case Kind::SubString:
				return 3;
		}
		Utils::Logger::LogError("Unknown parameter count for builtin function '" + m_name + "'");
	}

	std::string ToString() const override
	{
		return SignatureString();
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

private:
	Kind m_kind;
};

}
