#pragma once

#include <typeindex>
#include <typeinfo>

#include "Runtime/DataBuffers/DataBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

#include "Utils/Logger.h"

namespace Runtime {

class BufferUtils
{
public:

template<class T>
static const T *GetBuffer(const DataBuffer *buffer, bool assert = true)
{
	if (buffer->m_kind == T::BufferKind)
	{
		return static_cast<const T *>(buffer);
	}

	if (assert)
	{
		Utils::Logger::LogError("Cannot convert buffer " + buffer->Description() + " to kind " + DataBuffer::KindString(T::BufferKind));
	}
	return nullptr;
}

template<class T>
static T *GetBuffer(DataBuffer *buffer, bool assert = true)
{
	if (buffer->m_kind == T::BufferKind)
	{
		return static_cast<T *>(buffer);
	}

	if (assert)
	{
		Utils::Logger::LogError("Cannot convert buffer " + buffer->Description() + " to kind " + DataBuffer::KindString(T::BufferKind));
	}
	return nullptr;
}

template<class T>
static TypedVectorBuffer<T> *GetVectorBuffer(DataBuffer *buffer, bool assert = true)
{
	if (auto vectorBuffer = GetBuffer<VectorBuffer>(buffer, assert))
	{
		if (vectorBuffer->m_typeid == typeid(T))
		{
			return static_cast<TypedVectorBuffer<T> *>(vectorBuffer);
		}
	}

	if (assert)
	{
		Utils::Logger::LogError("Cannot convert buffer " + buffer->Description() + " to TypedVector");
	}
	return nullptr;
}

};

}
