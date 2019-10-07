#pragma once

namespace Analysis {

class ThreadGeometry
{
public:
	enum class Kind
	{
		Vector,
		List
	};

	ThreadGeometry(Kind kind, unsigned long size) : m_kind(kind), m_size(size) {}

	Kind GetKind() const { return m_kind; }
	unsigned long GetSize() const { return m_size; }

	std::string ToString() const
	{
		switch (m_kind)
		{
			case Kind::Vector:
				return "Vector<" + std::to_string(m_size) + ">";
			case Kind::List:
				return "List<" + std::to_string(m_size) + ">";
			default:
				return "<Unknown>";
		}
	}

private:
	Kind m_kind;
	unsigned long m_size = 0;
};

}
