#pragma once

#include <string>

#include "Runtime/DataBuffers/DataObjects/DataObject.h"

#include "Runtime/StringBucket.h"

#include "CUDA/Vector.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"
#include "HorseIR/Utils/TypeUtils.h"

#include "Utils/Date.h"

namespace Runtime {

class VectorData : public DataObject
{
public:
	static VectorData *CreateVector(const HorseIR::BasicType *type, unsigned long size);

	virtual int Compare(unsigned int i1, unsigned int i2) const = 0;

	virtual size_t GetElementCount() const = 0;
	virtual size_t GetElementSize() const = 0;

	virtual void Resize(size_t size) = 0;

	virtual std::string DebugDumpElement(unsigned int index, unsigned int indent = 0, bool preindent = false) const = 0;

	virtual void Clear() = 0;
};

template<typename T>
class TypedVectorData : public VectorData
{
public:
	TypedVectorData(const HorseIR::BasicType *elementType, const CUDA::Vector<T>& data) : m_type(elementType), m_data(data) {}
	TypedVectorData(const HorseIR::BasicType *elementType, CUDA::Vector<T>&& data) : m_type(elementType), m_data(std::move(data)) {}
	TypedVectorData(const HorseIR::BasicType *elementType, unsigned long size) : m_type(elementType), m_size(size)
	{
		m_data.resize(size);
	}

	const HorseIR::BasicType *GetType() const { return m_type; }

	int Compare(unsigned int i1, unsigned int i2) const override
	{
		if constexpr(std::is_same<T, std::uint64_t>::value)
		{
			if (HorseIR::TypeUtils::IsCharacterType(m_type))
			{
				const auto s1 = StringBucket::RecoverString(m_data[i1]);
				const auto s2 = StringBucket::RecoverString(m_data[i2]);
				return strcmp(s1, s2);
			}
		}

		auto& v1 = m_data[i1];
		auto& v2 = m_data[i2];

		if (v1 < v2)
		{
			return -1;
		}
		return (v1 > v2);
	}

	template<typename C>
	const C GetValue(unsigned int i) const
	{
		if constexpr(std::is_convertible<T, C>::value)
		{
			return static_cast<C>(m_data.at(i));
		}
		else
		{
			Utils::Logger::LogError("Unable to convert typed vector data to requested type");
		}
	}

	const CUDA::Vector<T>& GetValues() const { return m_data; }
	CUDA::Vector<T>& GetValues() { return m_data; }

	const T& GetValue(unsigned int i) const { return m_data.at(i); }
	void SetValue(unsigned int i, const T& value) { m_data.at(i) = value; }

        void *GetData() override { return m_data.data(); }
	size_t GetDataSize() const override { return m_data.size() * sizeof(T); }

	size_t GetElementCount() const override { return m_data.size(); }
	size_t GetElementSize() const override { return sizeof(T); }

	void Resize(size_t size)
	{
		m_data.resize(size);
		m_size = size;
	}

	std::string Description() const
	{
		return (HorseIR::PrettyPrinter::PrettyString(m_type) + "(" + std::to_string(GetElementSize()) + " bytes) x " + std::to_string(GetElementCount()));
	}

	std::string DebugDump(unsigned int indent = 0, bool preindent = false) const
	{
		std::string string;
		if (!preindent)
		{
			string += std::string(indent * Utils::Logger::IndentSize, ' ');
		}

		auto count = std::min(m_data.size(), 32ul);
		if (count != 1)
		{
			string += "(";
		}
		bool first = true;
		for (auto i = 0u; i < count; i++)
		{
			if (!first)
			{
				string += ", ";
			}
			first = false;
			string += _DebugDumpElement(i);
		}
		if (m_data.size() > count)
		{
			string += ", ... [" + std::to_string(m_data.size()) + "]";
		}
		if (count != 1)
		{
			string += ")";
		}
		string += ":" + HorseIR::PrettyPrinter::PrettyString(m_type);
		return string;
	}

	std::string DebugDumpElement(unsigned int index, unsigned int indent = 0, bool preindent = false) const override
	{
		if (preindent)
		{
			return _DebugDumpElement(index);
		}

		std::string indentString(indent * Utils::Logger::IndentSize, ' ');
		return indentString + _DebugDumpElement(index);
	}

	void Clear() override
	{
		m_data.clear();
		m_data.resize(m_size);
	}

private:
	std::string _DebugDumpElement(unsigned int index) const
	{
		if constexpr(std::is_pointer<T>::value)
		{
			std::stringstream stream;
			stream << *m_data.at(index);
			return stream.str();
		}
		else
		{
			switch (m_type->GetBasicKind())
			{
				case HorseIR::BasicType::BasicKind::Char:
				{
					return std::string(1, m_data.at(index));
				}
				case HorseIR::BasicType::BasicKind::Symbol:
				case HorseIR::BasicType::BasicKind::String:
				{
					return StringBucket::RecoverString(m_data.at(index));
				}
				case HorseIR::BasicType::BasicKind::Datetime:
				{
					auto [year, month, day, hour, minute, second, millisecond]
						= Utils::Date::DatetimeFromEpoch(m_data.at(index));
					HorseIR::DateValue date(year, month, day);
					HorseIR::TimeValue time(hour, minute, second, millisecond);
					HorseIR::DatetimeValue datetime(&date, &time);
					return datetime.ToString();
				}
				case HorseIR::BasicType::BasicKind::Date:
				{
					auto [year, month, day] = Utils::Date::DateFromEpoch(m_data.at(index));
					HorseIR::DateValue date(year, month, day);
					return date.ToString();
				}
				case HorseIR::BasicType::BasicKind::Month:
				{
					auto [year, month, day] = Utils::Date::DateFromEpoch(m_data.at(index));
					HorseIR::MonthValue date(month, day);
					return date.ToString();
				}
				case HorseIR::BasicType::BasicKind::Minute:
				{
					auto [hour, minute, second] = Utils::Date::TimeFromEpoch(m_data.at(index));
					HorseIR::MinuteValue time(hour, minute);
					return time.ToString();
				}
				case HorseIR::BasicType::BasicKind::Second:
				{
					auto [hour, minute, second] = Utils::Date::TimeFromEpoch(m_data.at(index));
					HorseIR::SecondValue time(hour, minute, second);
					return time.ToString();
				}
				case HorseIR::BasicType::BasicKind::Time:
				{
					auto [year, month, day, hour, minute, second, millisecond]
						= Utils::Date::DatetimeFromEpoch(m_data.at(index));
					HorseIR::TimeValue time(hour, minute, second, millisecond);
					return time.ToString();
				}
				default:
				{
					return std::to_string(m_data.at(index));
				}
			}
		}
	}

	const HorseIR::BasicType *m_type = nullptr;

	CUDA::Vector<T> m_data;
	unsigned int m_size = 0;
};

}
