#pragma once

#include "CUDA/Data.h"

namespace CUDA {

class Constant : public Data
{

};

template<class T>
class TypedConstant : public Constant
{
public:
	TypedConstant(const T& value) : m_value(value) {}

	const void *GetAddress() const override { return &m_value; }

	T GetValue() const { return m_value; }
	size_t GetSize() const { return sizeof(T); }

private:
	T m_value;
};

}
