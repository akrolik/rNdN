#pragma once

#include "Codegen/ExpressionGeneratorBase.h"

template<PTX::Bits B, class T>
class ExpressionGenerator : public ExpressionGeneratorBase<B, T> {
public:
	using ExpressionGeneratorBase<B, T>::ExpressionGeneratorBase;
};

template<PTX::Bits B>
class ExpressionGenerator<B, PTX::Int8Type> : public ExpressionGeneratorBase<B, PTX::Int8Type>
{
public:
	using ExpressionGeneratorBase<B, PTX::Int8Type>::ExpressionGeneratorBase;
	
	void GenerateAdd(const PTX::Operand<PTX::Int8Type> *src1, const PTX::Operand<PTX::Int8Type> *src2) override
	{
		std::cerr << "[ERROR] TODO, conversion from i8 to i16" << std::endl;
		// std::exit(EXIT_FAILURE);
	}

	void GenerateMove(const PTX::Operand<PTX::Int8Type> *src) override
	{
		std::cerr << "[ERROR] TODO, conversion from i8 to i16" << std::endl;
		// std::exit(EXIT_FAILURE);
	}
};
