#pragma once

#include "HorseIR/Tree/Module.h"
#include "HorseIR/Tree/BuiltinMethod.h"

namespace HorseIR {

static Module *BuiltinModule = new Module("Builtin", {
	// Unary
	new BuiltinMethod(BuiltinMethod::Kind::Absolute),
	new BuiltinMethod(BuiltinMethod::Kind::Negate),
	new BuiltinMethod(BuiltinMethod::Kind::Ceiling),
	new BuiltinMethod(BuiltinMethod::Kind::Floor),
	new BuiltinMethod(BuiltinMethod::Kind::Round),
	new BuiltinMethod(BuiltinMethod::Kind::Conjugate),
	new BuiltinMethod(BuiltinMethod::Kind::Reciprocal),
	new BuiltinMethod(BuiltinMethod::Kind::Sign),
	new BuiltinMethod(BuiltinMethod::Kind::Pi),
	new BuiltinMethod(BuiltinMethod::Kind::Not),
	new BuiltinMethod(BuiltinMethod::Kind::Logarithm),
	new BuiltinMethod(BuiltinMethod::Kind::Logarithm2),
	new BuiltinMethod(BuiltinMethod::Kind::Logarithm10),
	new BuiltinMethod(BuiltinMethod::Kind::SquareRoot),
	new BuiltinMethod(BuiltinMethod::Kind::Exponential),
	new BuiltinMethod(BuiltinMethod::Kind::Date),
	new BuiltinMethod(BuiltinMethod::Kind::DateYear),
	new BuiltinMethod(BuiltinMethod::Kind::DateMonth),
	new BuiltinMethod(BuiltinMethod::Kind::DateDay),
	new BuiltinMethod(BuiltinMethod::Kind::Time),
	new BuiltinMethod(BuiltinMethod::Kind::TimeHour),
	new BuiltinMethod(BuiltinMethod::Kind::TimeMinute),
	new BuiltinMethod(BuiltinMethod::Kind::TimeSecond),
	new BuiltinMethod(BuiltinMethod::Kind::TimeMillisecond),
	new BuiltinMethod(BuiltinMethod::Kind::Cosine),
	new BuiltinMethod(BuiltinMethod::Kind::Sine),
	new BuiltinMethod(BuiltinMethod::Kind::Tangent),
	new BuiltinMethod(BuiltinMethod::Kind::InverseCosine),
	new BuiltinMethod(BuiltinMethod::Kind::InverseSine),
	new BuiltinMethod(BuiltinMethod::Kind::InverseTangent),
	new BuiltinMethod(BuiltinMethod::Kind::HyperbolicCosine),
	new BuiltinMethod(BuiltinMethod::Kind::HyperbolicSine),
	new BuiltinMethod(BuiltinMethod::Kind::HyperbolicTangent),
	new BuiltinMethod(BuiltinMethod::Kind::HyperbolicInverseCosine),
	new BuiltinMethod(BuiltinMethod::Kind::HyperbolicInverseSine),
	new BuiltinMethod(BuiltinMethod::Kind::HyperbolicInverseTangent),

	// Binary
	new BuiltinMethod(BuiltinMethod::Kind::Less),
	new BuiltinMethod(BuiltinMethod::Kind::Greater),
	new BuiltinMethod(BuiltinMethod::Kind::LessEqual),
	new BuiltinMethod(BuiltinMethod::Kind::GreaterEqual),
	new BuiltinMethod(BuiltinMethod::Kind::Equal),
	new BuiltinMethod(BuiltinMethod::Kind::NotEqual),
	new BuiltinMethod(BuiltinMethod::Kind::Plus),
	new BuiltinMethod(BuiltinMethod::Kind::Minus),
	new BuiltinMethod(BuiltinMethod::Kind::Multiply),
	new BuiltinMethod(BuiltinMethod::Kind::Divide),
	new BuiltinMethod(BuiltinMethod::Kind::Power),
	new BuiltinMethod(BuiltinMethod::Kind::LogarithmBase),
	new BuiltinMethod(BuiltinMethod::Kind::Modulo),
	new BuiltinMethod(BuiltinMethod::Kind::And),
	new BuiltinMethod(BuiltinMethod::Kind::Or),
	new BuiltinMethod(BuiltinMethod::Kind::Nand),
	new BuiltinMethod(BuiltinMethod::Kind::Nor),
	new BuiltinMethod(BuiltinMethod::Kind::Xor),
	new BuiltinMethod(BuiltinMethod::Kind::DatetimeDifference),

	// Algebraic Unary
	new BuiltinMethod(BuiltinMethod::Kind::Unique),
	new BuiltinMethod(BuiltinMethod::Kind::String),
	new BuiltinMethod(BuiltinMethod::Kind::Length),
	new BuiltinMethod(BuiltinMethod::Kind::Range),
	new BuiltinMethod(BuiltinMethod::Kind::Factorial),
	new BuiltinMethod(BuiltinMethod::Kind::Random),
	new BuiltinMethod(BuiltinMethod::Kind::Seed),
	new BuiltinMethod(BuiltinMethod::Kind::Flip),
	new BuiltinMethod(BuiltinMethod::Kind::Reverse),
	new BuiltinMethod(BuiltinMethod::Kind::Where),
	new BuiltinMethod(BuiltinMethod::Kind::Group),

	// Algebraic Binary
	new BuiltinMethod(BuiltinMethod::Kind::Append),
	new BuiltinMethod(BuiltinMethod::Kind::Like),
	new BuiltinMethod(BuiltinMethod::Kind::Compress),
	new BuiltinMethod(BuiltinMethod::Kind::Random_k),
	new BuiltinMethod(BuiltinMethod::Kind::IndexOf),
	new BuiltinMethod(BuiltinMethod::Kind::Take),
	new BuiltinMethod(BuiltinMethod::Kind::Drop),
	new BuiltinMethod(BuiltinMethod::Kind::Order),
	new BuiltinMethod(BuiltinMethod::Kind::Member),
	new BuiltinMethod(BuiltinMethod::Kind::Vector),

	// Reduction
	new BuiltinMethod(BuiltinMethod::Kind::Count),
	new BuiltinMethod(BuiltinMethod::Kind::Sum),
	new BuiltinMethod(BuiltinMethod::Kind::Average),
	new BuiltinMethod(BuiltinMethod::Kind::Minimum),
	new BuiltinMethod(BuiltinMethod::Kind::Maximum),

	// List
	new BuiltinMethod(BuiltinMethod::Kind::Raze),
	new BuiltinMethod(BuiltinMethod::Kind::Enlist),
	new BuiltinMethod(BuiltinMethod::Kind::ToList),
	new BuiltinMethod(BuiltinMethod::Kind::Each),
	new BuiltinMethod(BuiltinMethod::Kind::EachItem),
	new BuiltinMethod(BuiltinMethod::Kind::EachLeft),
	new BuiltinMethod(BuiltinMethod::Kind::EachRight),
	new BuiltinMethod(BuiltinMethod::Kind::Match),
	new BuiltinMethod(BuiltinMethod::Kind::Outer),

	// Database
	new BuiltinMethod(BuiltinMethod::Kind::Enum),
	new BuiltinMethod(BuiltinMethod::Kind::Dictionary),
	new BuiltinMethod(BuiltinMethod::Kind::Table),
	new BuiltinMethod(BuiltinMethod::Kind::KeyedTable),
	new BuiltinMethod(BuiltinMethod::Kind::Keys),
	new BuiltinMethod(BuiltinMethod::Kind::Values),
	new BuiltinMethod(BuiltinMethod::Kind::Meta),
	new BuiltinMethod(BuiltinMethod::Kind::ColumnValue),
	new BuiltinMethod(BuiltinMethod::Kind::LoadTable),
	new BuiltinMethod(BuiltinMethod::Kind::Fetch),
	new BuiltinMethod(BuiltinMethod::Kind::DatetimeAdd),
	new BuiltinMethod(BuiltinMethod::Kind::DatetimeSubtract),

	// Indexing
	new BuiltinMethod(BuiltinMethod::Kind::Index),
	new BuiltinMethod(BuiltinMethod::Kind::IndexAssignment),

	// Other
	new BuiltinMethod(BuiltinMethod::Kind::LoadCSV),
	new BuiltinMethod(BuiltinMethod::Kind::Print),
	new BuiltinMethod(BuiltinMethod::Kind::Format),
	new BuiltinMethod(BuiltinMethod::Kind::Fill)
});

}
