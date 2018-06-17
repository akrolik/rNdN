#pragma once

#define UNIMPLEMENTED

#include "PTX/Operands/Operand.h"

template<class T>
class ExpressionGeneratorInterface
{
public:
	virtual void GenerateAdd(const PTX::Operand<T> *src1, const PTX::Operand<T> *src2)
	{
		Unimplemented(__FUNCTION__);
	}

	virtual void GenerateMove(const PTX::Operand<T> *src)
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
