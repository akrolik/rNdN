#pragma once

#include "PTX/Operands/Operand.h"

template<class T>
class ExpressionGeneratorInterface
{
public:
	virtual void GenerateAdd(const PTX::TypedOperand<T> *src1, const PTX::TypedOperand<T> *src2)
	{
		Unimplemented(__FUNCTION__);
	}

	virtual void GenerateMove(const PTX::TypedOperand<T> *src)
	{
		Unimplemented(__FUNCTION__);
	}

protected:

	void Unimplemented(std::string method)
	{
		std::cerr << "[ERROR] Unsupported type " << T::Name() << " in " << method << std::endl;
		std::exit(EXIT_FAILURE);
	}

};
