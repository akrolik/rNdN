#pragma once

namespace CUDA {

class Constant
{
public:
	virtual void *GetAddress() = 0;
};

template<class T>
class TypedConstant : public Constant
{
public:
	TypedConstant(T& value) : m_value(value) {}

	void *GetAddress() override { return &m_value; }

	T GetValue() const { return m_value; }
	size_t GetSize() const { return sizeof(T); }

private:
	T m_value;
};

}
