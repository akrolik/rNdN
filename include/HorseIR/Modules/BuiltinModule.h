#pragma once

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {

static Module *BuiltinModule = new LibraryModule("Builtin", {
	// Unary
	new BuiltinFunction(BuiltinFunction::Primitive::Absolute),
	new BuiltinFunction(BuiltinFunction::Primitive::Negate),
	new BuiltinFunction(BuiltinFunction::Primitive::Ceiling),
	new BuiltinFunction(BuiltinFunction::Primitive::Floor),
	new BuiltinFunction(BuiltinFunction::Primitive::Round),
	new BuiltinFunction(BuiltinFunction::Primitive::Conjugate),
	new BuiltinFunction(BuiltinFunction::Primitive::Reciprocal),
	new BuiltinFunction(BuiltinFunction::Primitive::Sign),
	new BuiltinFunction(BuiltinFunction::Primitive::Pi),
	new BuiltinFunction(BuiltinFunction::Primitive::Not),
	new BuiltinFunction(BuiltinFunction::Primitive::Logarithm),
	new BuiltinFunction(BuiltinFunction::Primitive::Logarithm2),
	new BuiltinFunction(BuiltinFunction::Primitive::Logarithm10),
	new BuiltinFunction(BuiltinFunction::Primitive::SquareRoot),
	new BuiltinFunction(BuiltinFunction::Primitive::Exponential),
	new BuiltinFunction(BuiltinFunction::Primitive::Cosine),
	new BuiltinFunction(BuiltinFunction::Primitive::Sine),
	new BuiltinFunction(BuiltinFunction::Primitive::Tangent),
	new BuiltinFunction(BuiltinFunction::Primitive::InverseCosine),
	new BuiltinFunction(BuiltinFunction::Primitive::InverseSine),
	new BuiltinFunction(BuiltinFunction::Primitive::InverseTangent),
	new BuiltinFunction(BuiltinFunction::Primitive::HyperbolicCosine),
	new BuiltinFunction(BuiltinFunction::Primitive::HyperbolicSine),
	new BuiltinFunction(BuiltinFunction::Primitive::HyperbolicTangent),
	new BuiltinFunction(BuiltinFunction::Primitive::HyperbolicInverseCosine),
	new BuiltinFunction(BuiltinFunction::Primitive::HyperbolicInverseSine),
	new BuiltinFunction(BuiltinFunction::Primitive::HyperbolicInverseTangent),

	// Binary
	new BuiltinFunction(BuiltinFunction::Primitive::Less),
	new BuiltinFunction(BuiltinFunction::Primitive::Greater),
	new BuiltinFunction(BuiltinFunction::Primitive::LessEqual),
	new BuiltinFunction(BuiltinFunction::Primitive::GreaterEqual),
	new BuiltinFunction(BuiltinFunction::Primitive::Equal),
	new BuiltinFunction(BuiltinFunction::Primitive::NotEqual),
	new BuiltinFunction(BuiltinFunction::Primitive::Plus),
	new BuiltinFunction(BuiltinFunction::Primitive::Minus),
	new BuiltinFunction(BuiltinFunction::Primitive::Multiply),
	new BuiltinFunction(BuiltinFunction::Primitive::Divide),
	new BuiltinFunction(BuiltinFunction::Primitive::Power),
	new BuiltinFunction(BuiltinFunction::Primitive::LogarithmBase),
	new BuiltinFunction(BuiltinFunction::Primitive::Modulo),
	new BuiltinFunction(BuiltinFunction::Primitive::And),
	new BuiltinFunction(BuiltinFunction::Primitive::Or),
	new BuiltinFunction(BuiltinFunction::Primitive::Nand),
	new BuiltinFunction(BuiltinFunction::Primitive::Nor),
	new BuiltinFunction(BuiltinFunction::Primitive::Xor),

	// Algebraic Unary
	new BuiltinFunction(BuiltinFunction::Primitive::Unique),
	new BuiltinFunction(BuiltinFunction::Primitive::Range),
	new BuiltinFunction(BuiltinFunction::Primitive::Factorial),
	new BuiltinFunction(BuiltinFunction::Primitive::Random),
	new BuiltinFunction(BuiltinFunction::Primitive::Seed),
	new BuiltinFunction(BuiltinFunction::Primitive::Flip),
	new BuiltinFunction(BuiltinFunction::Primitive::Reverse),
	new BuiltinFunction(BuiltinFunction::Primitive::Where),
	new BuiltinFunction(BuiltinFunction::Primitive::Group),

	// Algebraic Binary
	new BuiltinFunction(BuiltinFunction::Primitive::Append),
	new BuiltinFunction(BuiltinFunction::Primitive::Like),
	new BuiltinFunction(BuiltinFunction::Primitive::Compress),
	new BuiltinFunction(BuiltinFunction::Primitive::Random_k),
	new BuiltinFunction(BuiltinFunction::Primitive::IndexOf),
	new BuiltinFunction(BuiltinFunction::Primitive::Take),
	new BuiltinFunction(BuiltinFunction::Primitive::Drop),
	new BuiltinFunction(BuiltinFunction::Primitive::Order),
	new BuiltinFunction(BuiltinFunction::Primitive::Member),
	new BuiltinFunction(BuiltinFunction::Primitive::Vector),

	// Reduction
	new BuiltinFunction(BuiltinFunction::Primitive::Length),
	new BuiltinFunction(BuiltinFunction::Primitive::Sum),
	new BuiltinFunction(BuiltinFunction::Primitive::Average),
	new BuiltinFunction(BuiltinFunction::Primitive::Minimum),
	new BuiltinFunction(BuiltinFunction::Primitive::Maximum),

	// List
	new BuiltinFunction(BuiltinFunction::Primitive::List),
	new BuiltinFunction(BuiltinFunction::Primitive::Raze),
	new BuiltinFunction(BuiltinFunction::Primitive::ToList),
	new BuiltinFunction(BuiltinFunction::Primitive::Each),
	new BuiltinFunction(BuiltinFunction::Primitive::EachItem),
	new BuiltinFunction(BuiltinFunction::Primitive::EachLeft),
	new BuiltinFunction(BuiltinFunction::Primitive::EachRight),
	new BuiltinFunction(BuiltinFunction::Primitive::Match),

	// Date
	new BuiltinFunction(BuiltinFunction::Primitive::Date),
	new BuiltinFunction(BuiltinFunction::Primitive::DateYear),
	new BuiltinFunction(BuiltinFunction::Primitive::DateMonth),
	new BuiltinFunction(BuiltinFunction::Primitive::DateDay),
	new BuiltinFunction(BuiltinFunction::Primitive::Time),
	new BuiltinFunction(BuiltinFunction::Primitive::TimeHour),
	new BuiltinFunction(BuiltinFunction::Primitive::TimeMinute),
	new BuiltinFunction(BuiltinFunction::Primitive::TimeSecond),
	new BuiltinFunction(BuiltinFunction::Primitive::TimeMillisecond),
	new BuiltinFunction(BuiltinFunction::Primitive::DatetimeAdd),
	new BuiltinFunction(BuiltinFunction::Primitive::DatetimeSubtract),
	new BuiltinFunction(BuiltinFunction::Primitive::DatetimeDifference),

	// Database
	new BuiltinFunction(BuiltinFunction::Primitive::Enum),
	new BuiltinFunction(BuiltinFunction::Primitive::Dictionary),
	new BuiltinFunction(BuiltinFunction::Primitive::Table),
	new BuiltinFunction(BuiltinFunction::Primitive::KeyedTable),
	new BuiltinFunction(BuiltinFunction::Primitive::Keys),
	new BuiltinFunction(BuiltinFunction::Primitive::Values),
	new BuiltinFunction(BuiltinFunction::Primitive::Meta),
	new BuiltinFunction(BuiltinFunction::Primitive::Fetch),
	new BuiltinFunction(BuiltinFunction::Primitive::ColumnValue),
	new BuiltinFunction(BuiltinFunction::Primitive::LoadTable),
	new BuiltinFunction(BuiltinFunction::Primitive::JoinIndex),

	// Indexing
	new BuiltinFunction(BuiltinFunction::Primitive::Index),
	new BuiltinFunction(BuiltinFunction::Primitive::IndexAssignment),

	// Other
	new BuiltinFunction(BuiltinFunction::Primitive::LoadCSV),
	new BuiltinFunction(BuiltinFunction::Primitive::Print),
	new BuiltinFunction(BuiltinFunction::Primitive::Format),
	new BuiltinFunction(BuiltinFunction::Primitive::String),
	new BuiltinFunction(BuiltinFunction::Primitive::SubString)
});

}
