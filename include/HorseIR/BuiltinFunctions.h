#pragma once

#include <string>
#include <iostream>

#include "HorseIR/Tree/Module.h"
#include "HorseIR/Tree/BuiltinMethod.h"

namespace HorseIR {

static HorseIR::Module *BuiltinModule = new HorseIR::Module("Builtin", {
	// Unary
	new HorseIR::BuiltinMethod("abs"),
	new HorseIR::BuiltinMethod("neg"),
	new HorseIR::BuiltinMethod("ceil"),
	new HorseIR::BuiltinMethod("floor"),
	new HorseIR::BuiltinMethod("round"),
	new HorseIR::BuiltinMethod("conj"),
	new HorseIR::BuiltinMethod("recip"),
	new HorseIR::BuiltinMethod("signum"),
	new HorseIR::BuiltinMethod("pi"),
	new HorseIR::BuiltinMethod("not"),
	new HorseIR::BuiltinMethod("log"),
	new HorseIR::BuiltinMethod("exp"),
	new HorseIR::BuiltinMethod("cos"),
	new HorseIR::BuiltinMethod("sin"),
	new HorseIR::BuiltinMethod("tan"),
	new HorseIR::BuiltinMethod("acos"),
	new HorseIR::BuiltinMethod("asin"),
	new HorseIR::BuiltinMethod("atan"),
	new HorseIR::BuiltinMethod("cosh"),
	new HorseIR::BuiltinMethod("sinh"),
	new HorseIR::BuiltinMethod("tanh"),
	new HorseIR::BuiltinMethod("acosh"),
	new HorseIR::BuiltinMethod("asinh"),
	new HorseIR::BuiltinMethod("atanh"),

	// Binary
	new HorseIR::BuiltinMethod("lt"),
	new HorseIR::BuiltinMethod("gt"),
	new HorseIR::BuiltinMethod("leq"),
	new HorseIR::BuiltinMethod("geq"),
	new HorseIR::BuiltinMethod("eq"),
	new HorseIR::BuiltinMethod("neq"),
	new HorseIR::BuiltinMethod("plus"),
	new HorseIR::BuiltinMethod("minus"),
	new HorseIR::BuiltinMethod("mul"),
	new HorseIR::BuiltinMethod("div"),
	new HorseIR::BuiltinMethod("power"),
	new HorseIR::BuiltinMethod("log2"),
	new HorseIR::BuiltinMethod("mod"),
	new HorseIR::BuiltinMethod("and"),
	new HorseIR::BuiltinMethod("or"),
	new HorseIR::BuiltinMethod("nand"),
	new HorseIR::BuiltinMethod("nor"),
	new HorseIR::BuiltinMethod("xor"),

	// Binary Algebraic
	new HorseIR::BuiltinMethod("compress"),

	// Reduction
	new HorseIR::BuiltinMethod("count"),
	new HorseIR::BuiltinMethod("sum"),
	new HorseIR::BuiltinMethod("avg"),
	new HorseIR::BuiltinMethod("min"),
	new HorseIR::BuiltinMethod("max"),

	// List
	new HorseIR::BuiltinMethod("enlist"),

	// Database
	new HorseIR::BuiltinMethod("table"),
	new HorseIR::BuiltinMethod("column_value"),
	new HorseIR::BuiltinMethod("load_table"),

	// Other
	new HorseIR::BuiltinMethod("fill")
});

enum class BuiltinFunction
{
	Unsupported,
	Absolute,
	Negate,
	Ceiling,
	Floor,
	Round,
	Reciprocal,
	Conjugate,
	Sign,
	Pi,
	Not,
	Logarithm,
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
	Logarithm2,
	Modulo,
	And,
	Or,
	Nand,
	Nor,
	Xor,
	Compress,
	Count,
	Sum,
	Average,
	Minimum,
	Maximum,
	Fill
};

static BuiltinFunction GetBuiltinFunction(const std::string& name)
{	
	if (name == "abs")
	{
		return BuiltinFunction::Absolute;
	}
	else if (name == "neg")
	{
		return BuiltinFunction::Negate;
	}
	else if (name == "ceil")
	{
		return BuiltinFunction::Ceiling;
	}
	else if (name == "floor")
	{
		return BuiltinFunction::Floor;
	}
	else if (name == "round")
	{
		return BuiltinFunction::Round;
	}
	else if (name == "conj")
	{
		std::cerr << "[ERROR] Complex number functions are not supported" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	else if (name == "recip")
	{
		return BuiltinFunction::Reciprocal;
	}
	else if (name == "signum")
	{
		return BuiltinFunction::Sign;
	}
	else if (name == "pi")
	{
		return BuiltinFunction::Pi;
	}
	else if (name == "not")
	{
		return BuiltinFunction::Not;
	}
	else if (name == "log")
	{
		return BuiltinFunction::Logarithm;
	}
	else if (name == "exp")
	{
		return BuiltinFunction::Exponential;
	}
	else if (name == "cos")
	{
		return BuiltinFunction::Cosine;
	}
	else if (name == "sin")
	{
		return BuiltinFunction::Sine;
	}
	else if (name == "tan")
	{
		return BuiltinFunction::Tangent;
	}
	else if (name == "acos")
	{
		return BuiltinFunction::InverseCosine;
	}
	else if (name == "asin")
	{
		return BuiltinFunction::InverseSine;
	}
	else if (name == "atan")
	{
		return BuiltinFunction::InverseTangent;
	}
	else if (name == "cosh")
	{
		return BuiltinFunction::HyperbolicCosine;
	}
	else if (name == "sinh")
	{
		return BuiltinFunction::HyperbolicSine;
	}
	else if (name == "tanh")
	{
		return BuiltinFunction::HyperbolicTangent;
	}
	else if (name == "acosh")
	{
		return BuiltinFunction::HyperbolicInverseCosine;
	}
	else if (name == "asinh")
	{
		return BuiltinFunction::HyperbolicInverseSine;
	}
	else if (name == "atanh")
	{
		return BuiltinFunction::HyperbolicInverseTangent;
	}
	else if (name == "lt")
	{
		return BuiltinFunction::Less;
	}
	else if (name == "gt")
	{
		return BuiltinFunction::Greater;
	}
	else if (name == "leq")
	{
		return BuiltinFunction::LessEqual;
	}
	else if (name == "geq")
	{
		return BuiltinFunction::GreaterEqual;
	}
	else if (name == "eq")
	{
		return BuiltinFunction::Equal;
	}
	else if (name == "neq")
	{
		return BuiltinFunction::NotEqual;
	}
	else if (name == "plus")
	{
		return BuiltinFunction::Plus;
	}
	else if (name == "minus")
	{
		return BuiltinFunction::Minus;
	}
	else if (name == "mul")
	{
		return BuiltinFunction::Multiply;
	}
	else if (name == "div")
	{
		return BuiltinFunction::Divide;
	}
	else if (name == "power")
	{
		return BuiltinFunction::Power;
	}
	else if (name == "log2")
	{
		return BuiltinFunction::Logarithm2;
	}
	else if (name == "mod")
	{
		return BuiltinFunction::Modulo;
	}
	else if (name == "and")
	{
		return BuiltinFunction::And;
	}
	else if (name == "or")
	{
		return BuiltinFunction::Or;
	}
	else if (name == "nand")
	{
		return BuiltinFunction::Nand;
	}
	else if (name == "nor")
	{
		return BuiltinFunction::Nor;
	}
	else if (name == "xor")
	{
		return BuiltinFunction::Xor;
	}
	else if (name == "compress")
	{
		return BuiltinFunction::Compress;
	}
	else if (name == "count")
	{
		return BuiltinFunction::Count;
	}
	else if (name == "sum")
	{
		return BuiltinFunction::Sum;
	}
	else if (name == "avg")
	{
		return BuiltinFunction::Average;
	}
	else if (name == "min")
	{
		return BuiltinFunction::Minimum;
	}
	else if (name == "max")
	{
		return BuiltinFunction::Maximum;
	}
	else if (name == "fill")
	{
		return BuiltinFunction::Fill;
	}
	return BuiltinFunction::Unsupported;
}

}
