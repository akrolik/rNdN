#pragma once

#include <string>

#include "HorseIR/Tree/FunctionDeclaration.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

#include "Utils/Logger.h"

namespace HorseIR {

class BuiltinFunction : public FunctionDeclaration
{
public:
	enum class Primitive {
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

		// Algebraic Unary
		Unique,
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
		Replicate,
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
		Length,
		Sum,
		Average,
		Minimum,
		Maximum,

		// List
		Raze,
		List,
		ToList,
		Each,
		EachItem,
		EachLeft,
		EachRight,
		Match,

		// Date
		Date,
		DateYear,
		DateMonth,
		DateDay,
		Time,
		TimeHour,
		TimeMinute,
		TimeSecond,
		TimeMillisecond,
		DatetimeAdd,
		DatetimeSubtract,
		DatetimeDifference,

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
		JoinIndex,

		// Indexing
		Index,
		IndexAssignment,

		// Other
		LoadCSV,
		Print,
		String,
		SubString,

		// GPU
		GPUOrderLib,
		GPUOrderInit,
		GPUOrder,
		GPUOrderShared,

		GPUGroupLib,
		GPUGroup,

		GPUUniqueLib,
		GPUUnique,

		GPULoopJoinLib,
		GPULoopJoinCount,
		GPULoopJoin,

		GPUHashJoinLib,
		GPUHashJoinCreate,
		GPUHashJoinCount,
		GPUHashJoin,

		GPUHashMemberLib,
		GPUHashMemberCreate,
		GPUHashMember,

		GPULikeLib,
		GPULikeCacheLib
	};

	static const std::string PrimitiveName(Primitive primitive)
	{
		switch (primitive)
		{
			case Primitive::Absolute:
				return "abs";
			case Primitive::Negate:
				return "neg";
			case Primitive::Ceiling:
				return "ceil";
			case Primitive::Floor:
				return "floor";
			case Primitive::Round:
				return "round";
			case Primitive::Conjugate:
				return "conj";
			case Primitive::Reciprocal:
				return "recip";
			case Primitive::Sign:
				return "signum";
			case Primitive::Pi:
				return "pi";
			case Primitive::Not:
				return "not";
			case Primitive::Logarithm:
				return "log";
			case Primitive::Logarithm2:
				return "log2";
			case Primitive::Logarithm10:
				return "log10";
			case Primitive::SquareRoot:
				return "sqrt";
			case Primitive::Exponential:
				return "exp";
			case Primitive::Cosine:
				return "cos";
			case Primitive::Sine:
				return "sin";
			case Primitive::Tangent:
				return "tan";
			case Primitive::InverseCosine:
				return "acos";
			case Primitive::InverseSine:
				return "asin";
			case Primitive::InverseTangent:
				return "atan";
			case Primitive::HyperbolicCosine:
				return "cosh";
			case Primitive::HyperbolicSine:
				return "sinh";
			case Primitive::HyperbolicTangent:
				return "tanh";
			case Primitive::HyperbolicInverseCosine:
				return "acosh";
			case Primitive::HyperbolicInverseSine:
				return "asinh";
			case Primitive::HyperbolicInverseTangent:
				return "atanh";
			case Primitive::Less:
				return "lt";
			case Primitive::Greater:
				return "gt";
			case Primitive::LessEqual:
				return "leq";
			case Primitive::GreaterEqual:
				return "geq";
			case Primitive::Equal:
				return "eq";
			case Primitive::NotEqual:
				return "neq";
			case Primitive::Plus:
				return "plus";
			case Primitive::Minus:
				return "minus";
			case Primitive::Multiply:
				return "mul";
			case Primitive::Divide:
				return "div";
			case Primitive::Power:
				return "power";
			case Primitive::LogarithmBase:
				return "logb";
			case Primitive::Modulo:
				return "mod";
			case Primitive::And:
				return "and";
			case Primitive::Or:
				return "or";
			case Primitive::Nand:
				return "nand";
			case Primitive::Nor:
				return "nor";
			case Primitive::Xor:
				return "xor";
			case Primitive::Unique:
				return "unique";
			case Primitive::Range:
				return "range";
			case Primitive::Factorial:
				return "fact";
			case Primitive::Random:
				return "rand";
			case Primitive::Seed:
				return "seed";
			case Primitive::Flip:
				return "flip";
			case Primitive::Reverse:
				return "reverse";
			case Primitive::Where:
				return "where";
			case Primitive::Group:
				return "group";
			case Primitive::Append:
				return "append";
			case Primitive::Replicate:
				return "replicate";
			case Primitive::Like:
				return "like";
			case Primitive::Compress:
				return "compress";
			case Primitive::Random_k:
				return "randk";
			case Primitive::IndexOf:
				return "index_of";
			case Primitive::Take:
				return "take";
			case Primitive::Drop:
				return "drop";
			case Primitive::Order:
				return "order";
			case Primitive::Member:
				return "member";
			case Primitive::Vector:
				return "vector";
			case Primitive::Length:
				return "len";
			case Primitive::Sum:
				return "sum";
			case Primitive::Average:
				return "avg";
			case Primitive::Minimum:
				return "min";
			case Primitive::Maximum:
				return "max";
			case Primitive::Raze:
				return "raze";
			case Primitive::List:
				return "list";
			case Primitive::ToList:
				return "tolist";
			case Primitive::Each:
				return "each";
			case Primitive::EachItem:
				return "each_item";
			case Primitive::EachLeft:
				return "each_left";
			case Primitive::EachRight:
				return "each_right";
			case Primitive::Match:
				return "match";
			case Primitive::Date:
				return "date";
			case Primitive::DateYear:
				return "date_year";
			case Primitive::DateMonth:
				return "date_month";
			case Primitive::DateDay:
				return "date_day";
			case Primitive::Time:
				return "time";
			case Primitive::TimeHour:
				return "time_hour";
			case Primitive::TimeMinute:
				return "time_minute";
			case Primitive::TimeSecond:
				return "time_second";
			case Primitive::TimeMillisecond:
				return "time_mill";
			case Primitive::DatetimeAdd:
				return "datetime_add";
			case Primitive::DatetimeSubtract:
				return "datetime_sub";
			case Primitive::DatetimeDifference:
				return "datetime_diff";
			case Primitive::Enum:
				return "enum";
			case Primitive::Dictionary:
				return "dict";
			case Primitive::Table:
				return "table";
			case Primitive::KeyedTable:
				return "ktable";
			case Primitive::Keys:
				return "keys";
			case Primitive::Values:
				return "values";
			case Primitive::Meta:
				return "meta";
			case Primitive::Fetch:
				return "fetch";
			case Primitive::ColumnValue:
				return "column_value";
			case Primitive::LoadTable:
				return "load_table";
			case Primitive::JoinIndex:
				return "join_index";
			case Primitive::Index:
				return "index";
			case Primitive::IndexAssignment:
				return "index_a";
			case Primitive::LoadCSV:
				return "load_csv";
			case Primitive::Print:
				return "print";
			case Primitive::String:
				return "str";
			case Primitive::SubString:
				return "sub_string";
			case Primitive::GPUOrderLib:
				return "order_lib";
			case Primitive::GPUOrderInit:
				return "order_init";
			case Primitive::GPUOrder:
				return "order";
			case Primitive::GPUOrderShared:
				return "order_shared";
			case Primitive::GPUGroupLib:
				return "group_lib";
			case Primitive::GPUGroup:
				return "group";
			case Primitive::GPUUniqueLib:
				return "unique_lib";
			case Primitive::GPUUnique:
				return "unique";
			case Primitive::GPULoopJoinLib:
				return "loop_join_lib";
			case Primitive::GPULoopJoinCount:
				return "loop_join_count";
			case Primitive::GPULoopJoin:
				return "loop_join";
			case Primitive::GPUHashJoinLib:
				return "hash_join_lib";
			case Primitive::GPUHashJoinCreate:
				return "hash_join_create";
			case Primitive::GPUHashJoinCount:
				return "hash_join_count";
			case Primitive::GPUHashJoin:
				return "hash_join";
			case Primitive::GPUHashMemberLib:
				return "hash_member_lib";
			case Primitive::GPUHashMemberCreate:
				return "hash_member_create";
			case Primitive::GPUHashMember:
				return "hash_member";
			case Primitive::GPULikeLib:
				return "like_lib";
			case Primitive::GPULikeCacheLib:
				return "like_cache_lib";
		}
		return "<unknown>";
	}

	BuiltinFunction(Primitive primitive) : FunctionDeclaration(FunctionDeclaration::Kind::Builtin, PrimitiveName(primitive)), m_primitive(primitive) {}

	BuiltinFunction *Clone() const override
	{
		return new BuiltinFunction(m_primitive);
	}

	// Properties

	Primitive GetPrimitive() const { return m_primitive; }

	constexpr static int VariadicParameterCount = -1;

	int GetParameterCount() const
	{
		switch (m_primitive)
		{
			// Unary functions
			case Primitive::Absolute:
			case Primitive::Negate:
			case Primitive::Ceiling:
			case Primitive::Floor:
			case Primitive::Round:
			case Primitive::Conjugate:
			case Primitive::Reciprocal:
			case Primitive::Sign:
			case Primitive::Pi:
			case Primitive::Not:
			case Primitive::Logarithm:
			case Primitive::Logarithm2:
			case Primitive::Logarithm10:
			case Primitive::SquareRoot:
			case Primitive::Exponential:
			case Primitive::Cosine:
			case Primitive::Sine:
			case Primitive::Tangent:
			case Primitive::InverseCosine:
			case Primitive::InverseSine:
			case Primitive::InverseTangent:
			case Primitive::HyperbolicCosine:
			case Primitive::HyperbolicSine:
			case Primitive::HyperbolicTangent:
			case Primitive::HyperbolicInverseCosine:
			case Primitive::HyperbolicInverseSine:
			case Primitive::HyperbolicInverseTangent:
				return 1;

			// Binary functions
			case Primitive::Less:
			case Primitive::Greater:
			case Primitive::LessEqual:
			case Primitive::GreaterEqual:
			case Primitive::Equal:
			case Primitive::NotEqual:
			case Primitive::Plus:
			case Primitive::Minus:
			case Primitive::Multiply:
			case Primitive::Divide:
			case Primitive::Power:
			case Primitive::LogarithmBase:
			case Primitive::Modulo:
			case Primitive::And:
			case Primitive::Or:
			case Primitive::Nand:
			case Primitive::Nor:
			case Primitive::Xor:
				return 2;
			
			// Algebraic unary functions
			case Primitive::Unique:
			case Primitive::Range:
			case Primitive::Factorial:
			case Primitive::Random:
			case Primitive::Seed:
			case Primitive::Flip:
			case Primitive::Reverse:
			case Primitive::Where:
			case Primitive::Group:
				return 1;

			// Algebraic binary functions
			case Primitive::Append:
			case Primitive::Replicate:
			case Primitive::Like:
			case Primitive::Compress:
			case Primitive::Random_k:
			case Primitive::IndexOf:
			case Primitive::Take:
			case Primitive::Drop:
			case Primitive::Order:
			case Primitive::Member:
			case Primitive::Vector:
				return 2;

			// Reduction functions
			case Primitive::Length:
			case Primitive::Sum:
			case Primitive::Average:
			case Primitive::Minimum:
			case Primitive::Maximum:
				return 1;

			// List functions
			case Primitive::Raze:
				return 1;
			case Primitive::List:
				return VariadicParameterCount;
			case Primitive::ToList:
				return 1;
			case Primitive::Each:
				return 2;
			case Primitive::EachItem:
			case Primitive::EachLeft:
			case Primitive::EachRight:
				return 3;
			case Primitive::Match:
				return 2;

			// Date functions
			case Primitive::Date:
			case Primitive::DateYear:
			case Primitive::DateMonth:
			case Primitive::DateDay:
			case Primitive::Time:
			case Primitive::TimeHour:
			case Primitive::TimeMinute:
			case Primitive::TimeSecond:
			case Primitive::TimeMillisecond:
				return 1;
			case Primitive::DatetimeAdd:
			case Primitive::DatetimeSubtract:
				return 3;
			case Primitive::DatetimeDifference:
				return 2;

			// Database functions
			case Primitive::Enum:
			case Primitive::Dictionary:
			case Primitive::Table:
			case Primitive::KeyedTable:
				return 2;
			case Primitive::Keys:
				return 1;
			case Primitive::Values:
			case Primitive::Meta:
				return 1;
			case Primitive::ColumnValue:
				return 2;
			case Primitive::LoadTable:
				return 1;
			case Primitive::Fetch:
				return 1;
			case Primitive::JoinIndex:
				return VariadicParameterCount;

			// Indexing functions
			case Primitive::Index:
				return 2;
			case Primitive::IndexAssignment:
				return 3;

			// Others
			case Primitive::LoadCSV:
			case Primitive::Print:
			case Primitive::String:
				return 1;
			case Primitive::SubString:
				return 2;

			// GPU
			case Primitive::GPUOrderLib:
				return VariadicParameterCount; // @order_init, @order, @order_shared?, data, order?
			case Primitive::GPUOrderInit:
				return 2; // data, order
			case Primitive::GPUOrder:
			case Primitive::GPUOrderShared:
				return 3; // index, data, order
			case Primitive::GPUGroupLib:
				return VariadicParameterCount; // @order_init, @order, @order_shared?, @group, data
			case Primitive::GPUGroup:
				return 2; // index, data
			case Primitive::GPUUniqueLib:
				return VariadicParameterCount; // @order_init, @order, @order_shared?, @unique, data
			case Primitive::GPUUnique:
				return 2; // index, data
			case Primitive::GPULoopJoinLib:
				return 4; // @loop_join_count, @loop_join, data1, data2
			case Primitive::GPULoopJoinCount:
				return VariadicParameterCount; // @fn1, ..., @fnk, data1, data2
			case Primitive::GPULoopJoin:
				return VariadicParameterCount; // @fn1, ..., @fnk, data1, data2, offsets, count
			case Primitive::GPUHashJoinLib:
				return 5; // @hash_join_create, @hash_join_count, @hash_join, data1, data2
			case Primitive::GPUHashJoinCreate:
				return VariadicParameterCount; // @fn1, ..., @fnk, data2
			case Primitive::GPUHashJoinCount:
				return VariadicParameterCount; // @fn1, ..., @fnk, data1, hash_keys
			case Primitive::GPUHashJoin:
				return VariadicParameterCount; // @fn1, ..., @fnk, data1, hash_keys, hash_values, offsets, count
			case Primitive::GPUHashMemberLib:
				return 4; // @hash_create, @hash_member, data1, data2
			case Primitive::GPUHashMemberCreate:
				return 1; // data2
			case Primitive::GPUHashMember:
				return 2; // hash_keys, data1
			case Primitive::GPULikeLib:
			case Primitive::GPULikeCacheLib:
			{
				return 2;
			}

		}
		Utils::Logger::LogError("Unknown parameter count for builtin function '" + m_name + "'");
	}

	// Visitors

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

protected:
	Primitive m_primitive;
};

}
