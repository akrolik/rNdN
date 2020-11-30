#pragma once

namespace PTX {

template<class T, bool force = false, typename Enable = void>
class HalfModifier
{
public:
	constexpr static bool Enabled = false;
};

template<class T, bool force>
class HalfModifier<T, force,
      std::enable_if_t<REQUIRE_EXACT(T, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type) || force>>
{
public:
	constexpr static bool Enabled = true;

	enum class Half {
		Lower,
		Upper
	};

	HalfModifier(Half half) : m_half(half) {}

	Half GetHalf() const { return m_half; }
	void SetHalf(Half half) { m_half = half; }

	std::string OpCodeModifier() const
	{
		switch (m_half)
		{
			case Half::Lower:
				return ".lo";
			case Half::Upper:
				return ".hi";
		}
		return "";
	}

	bool IsActive() const
	{
		return true;
	}

protected:
	Half m_half;
};

}
