#pragma once

#include "Runtime/DataObject.h"

#include <string>
#include <vector>

#include "HorseIR/Tree/Types/Type.h"

namespace Runtime {

class Vector : public DataObject
{
public:
	virtual const HorseIR::Type *GetType() const = 0;
	virtual void *GetData() = 0;
	virtual size_t GetDataSize() const = 0;
};

template<typename T>
class TypedVector : public Vector
{
public:
	TypedVector(const HorseIR::Type *type, const std::vector<T>& data) : m_data(data) {}

	const HorseIR::Type *GetType() const override { return m_type; }
	void *GetData() override { return m_data.data(); }
	size_t GetDataSize() const override { return sizeof(T); }

private:
	const HorseIR::Type *m_type = nullptr;
	std::vector<T> m_data;
};

}
